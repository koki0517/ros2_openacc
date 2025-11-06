#include "rclcpp/rclcpp.hpp"
#include <iostream>
#include <vector>
#include "openacc.h" // Include OpenACC API header

// Helper function to get device type as a string
const char* get_acc_device_string(acc_device_t dev_type) {
    switch (dev_type) {
        case acc_device_none: return "None";
        case acc_device_default: return "Default";
        case acc_device_host: return "Host CPU";
        case acc_device_not_host: return "Not Host (generic)";
        case acc_device_nvidia: return "NVIDIA GPU";
        default: return "Unknown";
    }
}

class TestOpenACCNode : public rclcpp::Node
{
public:
  TestOpenACCNode()
  : Node("test_openacc_node")
  {
    // --- OpenACC Device Diagnostics ---
    RCLCPP_INFO(this->get_logger(), "--- Checking OpenACC Devices ---");
    acc_device_t initial_dev_type = acc_get_device_type();
    RCLCPP_INFO(this->get_logger(), "Initial device: %s", get_acc_device_string(initial_dev_type));

    int num_devices = acc_get_num_devices(acc_device_not_host);
    RCLCPP_INFO(this->get_logger(), "Found %d accelerator device(s).", num_devices);

    if (num_devices == 0) {
        RCLCPP_WARN(this->get_logger(), "No accelerator found. OpenACC will run on the host CPU.");
    }
    RCLCPP_INFO(this->get_logger(), "------------------------------------");
    // --- End Diagnostics ---


    // A simple loop with an OpenACC pragma
    int n = 10000000;
    std::vector<float> a(n), b(n);

    // Get raw pointers from vectors, which is more robust for OpenACC
    float* p_a = a.data();
    float* p_b = b.data();

    // Use an explicit data region to manage data on the device across kernels
    // create(p_a, p_b): Allocate memory on the device, but don't copy anything yet
    #pragma acc data create(p_a[0:n], p_b[0:n])
    {
      // First kernel: Initialize data on the device
      #pragma acc parallel loop
      for (int i=0; i<n; ++i) {
        p_a[i] = 0.0f;
        p_b[i] = 1.0f;
      }

      // Second kernel: Perform computation on the device
      #pragma acc parallel loop
      for (int i=0; i<n; ++i) {
        p_a[i] += p_b[i];
      }

      // Explicitly copy the result (only the first element) back to the host for printing
      #pragma acc update host(p_a[0:1])
    }

    RCLCPP_INFO(this->get_logger(), "OpenACC pragmas were included in the code.");
    RCLCPP_INFO(this->get_logger(), "The value of a[0] is: %2.2f", p_a[0]);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TestOpenACCNode>());
  rclcpp::shutdown();
  return 0;
}
