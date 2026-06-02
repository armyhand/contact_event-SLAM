cmake_minimum_required(VERSION 3.10)
project(haptic)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# 正确指向 setup.py 所声明的 python 包
ament_python_install_package(${PROJECT_NAME})
# 安装 launch 目录
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

