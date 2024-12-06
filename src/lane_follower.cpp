#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/Twist.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <deque>

class LaneFollower {
public:
    LaneFollower() : 
        nh_("~"), 
        look_ahead_distance_(100.0), // 픽셀 단위로 설정
        num_cols_(3),
        kalman_initialized_(false),
        frame_count_(0),
        display_interval_(5), // 5 프레임마다 디버그 표시
        window_size_(5), // 최근 5 프레임 평균
        expected_lane_width_(300.0) // 예상 차선 너비 (픽셀 단위)
    {
        image_sub_ = nh_.subscribe("/gmsl_camera/dev/video1/compressed", 1, &LaneFollower::imageCallback, this);
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

        initializeKalmanFilter();

        // 원본 해상도(1920x1080)에 기반한 ROI 포인트을 640x480으로 스케일링
        // 가로: 640/1920 = 1/3, 세로: 480/1080 ≈ 0.444
        roi_points_.push_back(cv::Point(static_cast<int>(401 / 3.0), static_cast<int>(1044 / 2.25)));
        roi_points_.push_back(cv::Point(static_cast<int>(910 / 3.0), static_cast<int>(654 / 2.25)));
        roi_points_.push_back(cv::Point(static_cast<int>(1064 / 3.0), static_cast<int>(655 / 2.25)));
        roi_points_.push_back(cv::Point(static_cast<int>(1445 / 3.0), static_cast<int>(1051 / 2.25)));
    }

    void spin() {
        ros::spin();
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Publisher cmd_vel_pub_;
    geometry_msgs::Twist twist_;
    double look_ahead_distance_;
    int window_size_;

    std::vector<cv::Point> roi_points_;
    cv::Mat previous_frame_with_lines_;

    std::vector<cv::Mat> debug_images_;
    int num_cols_;

    cv::KalmanFilter kf_;
    bool kalman_initialized_;

    int frame_count_;
    int display_interval_;
    double expected_lane_width_;

    struct LaneParams {
        double left_slope;
        double left_intercept;
        double right_slope;
        double right_intercept;
        bool valid;
    };
    std::deque<LaneParams> lane_history_;

    void initializeKalmanFilter() {
        int stateSize = 4; 
        int measSize = 4;  
        int contrSize = 0;

        kf_.init(stateSize, measSize, contrSize, CV_32F);

        // 상태 전이 행렬 설정 (left_slope, left_intercept, right_slope, right_intercept)
        cv::setIdentity(kf_.transitionMatrix);

        // 측정 행렬 설정
        kf_.measurementMatrix = cv::Mat::eye(measSize, stateSize, CV_32F);

        // 프로세스 잡음 공분산 행렬 Q
        cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));

        // 관측 잡음 공분산 행렬 R
        cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));

        // 오류 공분산 행렬 초기값
        cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
    }

    void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
        debug_images_.clear(); // 디버그 이미지 초기화

        // 압축 이미지를 OpenCV Mat으로 디코딩
        cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if(frame.empty()) {
            ROS_ERROR("이미지 디코딩 실패");
            return;
        }

        // 해상도 축소: 640x480
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(640, 480));
        debug_images_.push_back(resized_frame.clone());

        // ROI 영역 표시용 이미지
        cv::Mat roi_visual = resized_frame.clone();
        std::vector<std::vector<cv::Point>> roi_contours;
        roi_contours.push_back(roi_points_);
        cv::polylines(roi_visual, roi_contours, true, cv::Scalar(0,0,255), 2);
        debug_images_.push_back(roi_visual.clone());

        // 그레이스케일 변환
        cv::Mat gray;
        cv::cvtColor(resized_frame, gray, cv::COLOR_BGR2GRAY);
        debug_images_.push_back(gray.clone());

        // Gaussian Blur
        cv::Mat blur_img;
        cv::GaussianBlur(gray, blur_img, cv::Size(5, 5), 0);
        debug_images_.push_back(blur_img.clone());

        // Canny Edge Detection
        cv::Mat edges;
        cv::Canny(blur_img, edges, 50, 150); // 파라미터 조정
        debug_images_.push_back(edges.clone());

        // ROI 마스크 적용
        cv::Mat mask = cv::Mat::zeros(edges.size(), edges.type());
        std::vector<std::vector<cv::Point>> roi_contours2;
        roi_contours2.push_back(roi_points_);
        cv::fillPoly(mask, roi_contours2, cv::Scalar(255));
        cv::Mat masked_edges;
        cv::bitwise_and(edges, mask, masked_edges);
        debug_images_.push_back(masked_edges.clone());

        // 허프 변환: 차선 검출
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(masked_edges, lines, 1, CV_PI/180, 50, 50, 30); // 파라미터 조정
        cv::Mat line_image = cv::Mat::zeros(resized_frame.size(), resized_frame.type());
        if(!lines.empty()) {
            for(const auto& l : lines) {
                cv::line(line_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,255,0), 2);
            }
            debug_images_.push_back(line_image.clone());
        } else {
            ROS_WARN("허프 변환 선 검출 실패");
            debug_images_.push_back(line_image.clone());
        }

        std::vector<cv::Point> path;
        cv::Mat frame_with_lines = resized_frame.clone();
        bool lane_detected = false;
        double left_slope, left_intercept, right_slope, right_intercept;
        if(!lines.empty()) {
            lane_detected = getLaneCenter(lines, resized_frame, path, frame_with_lines,
                                          left_slope, left_intercept, right_slope, right_intercept);
        }

        // 현재 프레임 파라미터 저장
        LaneParams current_params;
        current_params.valid = lane_detected;
        if(lane_detected) {
            current_params.left_slope = left_slope;
            current_params.left_intercept = left_intercept;
            current_params.right_slope = right_slope;
            current_params.right_intercept = right_intercept;
        }

        lane_history_.push_back(current_params);
        if(static_cast<int>(lane_history_.size()) > window_size_) {
            lane_history_.pop_front();
        }

        LaneParams avg_params = getAverageParams();

        // avg_params를 이용해 차선 그리기
        if(avg_params.valid) {
            drawLane(frame_with_lines, avg_params);
            if(!path.empty()) {
                cv::Point look_ahead_point;
                if(getLookaheadPoint(path, look_ahead_distance_, frame_with_lines, look_ahead_point)) {
                    double steering_angle = computeSteeringAngle(look_ahead_point, resized_frame.size());
                    controlRobot(steering_angle);
                }
            }
            previous_frame_with_lines_ = frame_with_lines.clone();
            debug_images_.push_back(frame_with_lines.clone());
        } else {
            // 평균 결과도 유효하지 않을 경우
            ROS_WARN("평균 후에도 차선 유효하지 않음");
            if(!previous_frame_with_lines_.empty()) {
                debug_images_.push_back(previous_frame_with_lines_.clone());
            } else {
                debug_images_.push_back(resized_frame.clone());
            }
        }

        frame_count_++;
        if(frame_count_ % display_interval_ == 0) {
            displayImagesInGrid(debug_images_, "Debug Images");
        }
    } // Added closing brace for imageCallback

    bool getLaneCenter(const std::vector<cv::Vec4i>& lines, const cv::Mat& frame, 
                       std::vector<cv::Point>& path, cv::Mat& frame_with_lines,
                       double &left_slope, double &left_intercept, double &right_slope, double &right_intercept) 
    {
        std::vector<std::pair<double,double>> left_lines;  
        std::vector<std::pair<double,double>> right_lines;

        for(const auto& l : lines) {
            int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            if((x2 - x1) == 0) continue;
            double slope = static_cast<double>(y2 - y1) / static_cast<double>(x2 - x1);
            double intercept = y1 - slope * x1;

            // 기울기 범위 필터링
            if(slope < -0.5 && slope > -1.5) {
                left_lines.emplace_back(std::make_pair(slope, intercept));
            } else if(slope > 0.5 && slope < 1.5) {
                right_lines.emplace_back(std::make_pair(slope, intercept));
            }
        }

        auto avg_line = [](const std::vector<std::pair<double,double>>& lines) -> std::pair<double, double> {
            if(lines.empty()) {
                return std::make_pair(0.0, 0.0);
            }
            double slope_sum = 0.0, intercept_sum = 0.0;
            for(const auto& ln : lines) {
                slope_sum += ln.first;
                intercept_sum += ln.second;
            }
            double count = static_cast<double>(lines.size());
            return std::make_pair(slope_sum / count, intercept_sum / count);
        };

        std::pair<double, double> left_fit = avg_line(left_lines);
        std::pair<double, double> right_fit = avg_line(right_lines);

        bool has_left = !left_lines.empty();
        bool has_right = !right_lines.empty();

        ROS_INFO("왼쪽 라인 평균: %s, 오른쪽 라인 평균: %s", 
                 has_left ? "검출됨" : "없음", 
                 has_right ? "검출됨" : "없음");

        int height = frame.rows;
        int y1 = height;
        int y2 = static_cast<int>(height * 0.6);

        if(has_left && has_right) {
            left_slope = left_fit.first;
            left_intercept = left_fit.second;
            right_slope = right_fit.first;
            right_intercept = right_fit.second;

            // 칼만 필터 초기화
            if(!kalman_initialized_) {
                kf_.statePost.at<float>(0) = static_cast<float>(left_slope);
                kf_.statePost.at<float>(1) = static_cast<float>(left_intercept);
                kf_.statePost.at<float>(2) = static_cast<float>(right_slope);
                kf_.statePost.at<float>(3) = static_cast<float>(right_intercept);
                kalman_initialized_ = true;
            }

            // 칼만 필터 예측 및 수정
            cv::Mat prediction = kf_.predict();

            cv::Mat measurement(4, 1, CV_32F);
            measurement.at<float>(0) = static_cast<float>(left_slope);
            measurement.at<float>(1) = static_cast<float>(left_intercept);
            measurement.at<float>(2) = static_cast<float>(right_slope);
            measurement.at<float>(3) = static_cast<float>(right_intercept);

            cv::Mat estimated = kf_.correct(measurement);

            double est_left_slope = estimated.at<float>(0);
            double est_left_intercept = estimated.at<float>(1);
            double est_right_slope = estimated.at<float>(2);
            double est_right_intercept = estimated.at<float>(3);

            // 선의 x 좌표 계산
            int left_x1 = static_cast<int>((y1 - est_left_intercept) / est_left_slope);
            int left_x2 = static_cast<int>((y2 - est_left_intercept) / est_left_slope);
            int right_x1 = static_cast<int>((y1 - est_right_intercept) / est_right_slope);
            int right_x2 = static_cast<int>((y2 - est_right_intercept) / est_right_slope);

            // 중앙선 계산
            int mid_x1 = (left_x1 + right_x1) / 2;
            int mid_x2 = (left_x2 + right_x2) / 2;

            // 경로 설정
            path.clear();
            path.emplace_back(cv::Point(mid_x1, y1));
            path.emplace_back(cv::Point(mid_x2, y2));

            // 차선 그리기
            cv::line(frame_with_lines, cv::Point(left_x1, y1), cv::Point(left_x2, y2), cv::Scalar(255,0,0), 5);
            cv::line(frame_with_lines, cv::Point(right_x1, y1), cv::Point(right_x2, y2), cv::Scalar(255,0,0), 5);
            cv::line(frame_with_lines, cv::Point(mid_x1, y1), cv::Point(mid_x2, y2), cv::Scalar(0,255,0), 5);

            return true;
        } 
        else if(has_left || has_right) {
            // 한쪽 차선만 검출된 경우
            if(has_left) {
                left_slope = left_fit.first;
                left_intercept = left_fit.second;
                
                // 오른쪽 차선을 추정하지 않고, 이전 프레임의 오른쪽 차선을 유지
                double est_right_slope = right_slope;
                double est_right_intercept = right_intercept;

                if(!right_lines.empty()) {
                    est_right_slope = right_fit.first;
                    est_right_intercept = right_fit.second;
                } else if(!lane_history_.empty()) {
                    // 이전 프레임에서 오른쪽 차선이 있었는지 확인
                    for(auto it = lane_history_.rbegin(); it != lane_history_.rend(); ++it) {
                        if(it->valid) {
                            est_right_slope = it->right_slope;
                            est_right_intercept = it->right_intercept;
                            break;
                        }
                    }
                }

                // 중앙선 계산
                int left_x1 = static_cast<int>((y1 - left_intercept) / left_slope);
                int left_x2 = static_cast<int>((y2 - left_intercept) / left_slope);
                int right_x1 = static_cast<int>((y1 - est_right_intercept) / est_right_slope);
                int right_x2 = static_cast<int>((y2 - est_right_intercept) / est_right_slope);

                int mid_x1 = (left_x1 + right_x1) / 2;
                int mid_x2 = (left_x2 + right_x2) / 2;

                // 경로 설정
                path.clear();
                path.emplace_back(cv::Point(mid_x1, y1));
                path.emplace_back(cv::Point(mid_x2, y2));

                // 칼만 필터 초기화 및 업데이트
                if(!kalman_initialized_) {
                    kf_.statePost.at<float>(0) = static_cast<float>(left_slope);
                    kf_.statePost.at<float>(1) = static_cast<float>(left_intercept);
                    kf_.statePost.at<float>(2) = static_cast<float>(est_right_slope);
                    kf_.statePost.at<float>(3) = static_cast<float>(est_right_intercept);
                    kalman_initialized_ = true;
                }

                // 칼만 필터 예측 및 수정
                cv::Mat prediction = kf_.predict();

                cv::Mat measurement(4, 1, CV_32F);
                measurement.at<float>(0) = static_cast<float>(left_slope);
                measurement.at<float>(1) = static_cast<float>(left_intercept);
                measurement.at<float>(2) = static_cast<float>(est_right_slope);
                measurement.at<float>(3) = static_cast<float>(est_right_intercept);

                cv::Mat estimated = kf_.correct(measurement);

                double est_left_slope = estimated.at<float>(0);
                double est_left_intercept = estimated.at<float>(1);
                double est_right_slope_updated = estimated.at<float>(2);
                double est_right_intercept_updated = estimated.at<float>(3);

                // 선의 x 좌표 계산
                left_x1 = static_cast<int>((y1 - est_left_intercept) / est_left_slope);
                left_x2 = static_cast<int>((y2 - est_left_intercept) / est_left_slope);
                right_x1 = static_cast<int>((y1 - est_right_intercept_updated) / est_right_slope);
                right_x2 = static_cast<int>((y2 - est_right_intercept_updated) / est_right_slope);

                // 중앙선 계산
                mid_x1 = (left_x1 + right_x1) / 2;
                mid_x2 = (left_x2 + right_x2) / 2;

                // 차선 그리기
                cv::line(frame_with_lines, cv::Point(left_x1, y1), cv::Point(left_x2, y2), cv::Scalar(255,0,0), 5);
                cv::line(frame_with_lines, cv::Point(right_x1, y1), cv::Point(right_x2, y2), cv::Scalar(255,0,0), 5);
                cv::line(frame_with_lines, cv::Point(mid_x1, y1), cv::Point(mid_x2, y2), cv::Scalar(0,255,0), 5);

                return true;
            }
            return false;
        }

        return false; // Added return statement to ensure all control paths return a value
    } // Added closing brace for getLaneCenter

    LaneParams getAverageParams() {
        double l_slope_sum = 0.0, l_int_sum = 0.0, r_slope_sum = 0.0, r_int_sum = 0.0;
        int count = 0;
        for(auto &lp : lane_history_) {
            if(lp.valid) {
                l_slope_sum += lp.left_slope;
                l_int_sum += lp.left_intercept;
                r_slope_sum += lp.right_slope;
                r_int_sum += lp.right_intercept;
                count++;
            }
        }
        LaneParams avg;
        avg.valid = (count > 0);
        if(avg.valid) {
            avg.left_slope = l_slope_sum / count;
            avg.left_intercept = l_int_sum / count;
            avg.right_slope = r_slope_sum / count;
            avg.right_intercept = r_int_sum / count;
        }
        return avg;
    }

    void drawLane(cv::Mat& frame_with_lines, const LaneParams& params) {
        if(!params.valid) return;
        int height = frame_with_lines.rows;
        int y1 = height;
        int y2 = static_cast<int>(height * 0.6);

        double est_left_slope = params.left_slope;
        double est_left_intercept = params.left_intercept;
        double est_right_slope = params.right_slope;
        double est_right_intercept = params.right_intercept;

        int left_x1 = static_cast<int>((y1 - est_left_intercept) / est_left_slope);
        int left_x2 = static_cast<int>((y2 - est_left_intercept) / est_left_slope);
        int right_x1 = static_cast<int>((y1 - est_right_intercept) / est_right_slope);
        int right_x2 = static_cast<int>((y2 - est_right_intercept) / est_right_slope);

        int mid_x1 = (left_x1 + right_x1) / 2;
        int mid_x2 = (left_x2 + right_x2) / 2;

        // 파란색(255,0,0) 차선선 두께 5
        cv::line(frame_with_lines, cv::Point(left_x1, y1), cv::Point(left_x2, y2), cv::Scalar(255,0,0), 5);
        cv::line(frame_with_lines, cv::Point(right_x1, y1), cv::Point(right_x2, y2), cv::Scalar(255,0,0), 5);

        // 중앙선 녹색(0,255,0) 두께 5
        cv::line(frame_with_lines, cv::Point(mid_x1, y1), cv::Point(mid_x2, y2), cv::Scalar(0,255,0), 5);
    }

    bool getLookaheadPoint(const std::vector<cv::Point>& path, double look_ahead_distance, 
                           cv::Mat& frame_with_lines, cv::Point& look_ahead_point)
    {
        if(path.size() < 2) return false;
        cv::Point start = path[0];
        cv::Point end = path[1];

        double dx = end.x - start.x;
        double dy = static_cast<double>(start.y - end.y); // 이미지 좌표계 고려

        double distance = std::sqrt(dx*dx + dy*dy);
        if(distance == 0.0) {
            ROS_WARN("거리 계산 오류: 거리 = 0");
            return false;
        }

        double ratio = look_ahead_distance / distance;
        if(ratio > 1.0) ratio = 1.0;

        int look_x = static_cast<int>(start.x + dx * ratio);
        int look_y = static_cast<int>(start.y - dy * ratio);

        if(look_x >= 0 && look_x < frame_with_lines.cols && look_y >=0 && look_y < frame_with_lines.rows) {
            // 미리보기 포인트 노란색 원
            cv::circle(frame_with_lines, cv::Point(look_x, look_y), 10, cv::Scalar(0,255,255), -1);
        } else {
            ROS_WARN("타겟 포인트가 이미지 범위를 벗어났습니다.");
        }

        look_ahead_point = cv::Point(look_x, look_y);
        return true;
    }

    double computeSteeringAngle(const cv::Point& look_ahead_point, const cv::Size& frame_size) {
        double camera_center = static_cast<double>(frame_size.width) / 2.0;
        double dx = static_cast<double>(look_ahead_point.x) - camera_center;
        double dy = static_cast<double>(frame_size.height) - look_ahead_point.y;
        if(dy == 0) dy = 0.0001;

        double angle = std::atan2(dx, dy);
        double angle_deg = angle * 180.0 / M_PI;

        double steering_angle = angle_deg / 45.0; // 최대 45도 범위 가정
        if(steering_angle > 1.0) steering_angle = 1.0;
        if(steering_angle < -1.0) steering_angle = -1.0;

        return steering_angle;
    }

    void controlRobot(double steering_angle) {
        geometry_msgs::Twist twist;
        twist.linear.x = 0.2; // 전진 속도
        twist.angular.z = steering_angle; // 조향 각도
        cmd_vel_pub_.publish(twist);
        ROS_INFO("Cmd_vel published: linear.x=%.2f, angular.z=%.2f", twist.linear.x, twist.angular.z);
    }

    void displayImagesInGrid(const std::vector<cv::Mat>& images, const std::string& window_name) {
        if(images.empty()) return;

        int num_images = static_cast<int>(images.size());
        int cols = num_cols_;
        int rows = (num_images + cols - 1) / cols;

        int max_width = 0; 
        int max_height = 0;
        for (const auto& img : images) {
            if(img.cols > max_width) max_width = img.cols;
            if(img.rows > max_height) max_height = img.rows;
        }

        std::vector<cv::Mat> resized_images;
        resized_images.reserve(num_images);
        for (const auto& img : images) {
            cv::Mat disp_img;
            if(img.channels() == 1) {
                cv::cvtColor(img, disp_img, cv::COLOR_GRAY2BGR);
            } else {
                disp_img = img.clone();
            }
            int right_pad = max_width - disp_img.cols;
            int bottom_pad = max_height - disp_img.rows;
            cv::copyMakeBorder(disp_img, disp_img, 0, bottom_pad, 0, right_pad, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            resized_images.push_back(disp_img);
        }

        std::vector<cv::Mat> row_images;
        row_images.reserve(rows);
        for(int r = 0; r < rows; r++) {
            int start_idx = r * cols;
            int end_idx = std::min(start_idx + cols, num_images);
            std::vector<cv::Mat> row_imgs;
            row_imgs.reserve(cols);
            for(int c = start_idx; c < end_idx; c++) {
                row_imgs.push_back(resized_images[c]);
            }
            while(static_cast<int>(row_imgs.size()) < cols) {
                cv::Mat blank = cv::Mat::zeros(max_height, max_width, CV_8UC3);
                row_imgs.push_back(blank);
            }
            cv::Mat row_concat;
            cv::hconcat(row_imgs, row_concat);
            row_images.push_back(row_concat);
        }

        cv::Mat grid_image;
        cv::vconcat(row_images, grid_image);

        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::imshow(window_name, grid_image);
        cv::waitKey(1);
    }
}; // 클래스 정의 종료

int main(int argc, char** argv) {
    ros::init(argc, argv, "lane_follower");
    LaneFollower lf;
    lf.spin();
    return 0;
}
