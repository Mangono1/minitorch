#pragma once

#include "minitorch/vulkan/buffer.h"
#include "minitorch/vulkan/compute.h"
#include "minitorch/vulkan/context.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace minitorch {

class VulkanComputePipeline {
public:
    VulkanComputePipeline(
        VulkanContext& context,
        const std::string& shader_path,
        std::uint32_t binding_count = 2
    );

    ~VulkanComputePipeline();

    VulkanComputePipeline(
        const VulkanComputePipeline&
    ) = delete;

    VulkanComputePipeline& operator=(
        const VulkanComputePipeline&
    ) = delete;

    VulkanComputePipeline(
        VulkanComputePipeline&& other
    ) noexcept;

    VulkanComputePipeline& operator=(
        VulkanComputePipeline&& other
    ) noexcept;

    bool valid() const;

    void dispatch(
        VulkanCompute& compute,
        const std::vector<VulkanBuffer*>& buffers,
        std::size_t count
    );

    void dispatch(
        VulkanCompute& compute,
        const VulkanBuffer& input,
        VulkanBuffer& output,
        std::size_t count
    );

    void dispatch_scalar(
        VulkanCompute& compute,
        const VulkanBuffer& input,
        VulkanBuffer& output,
        std::size_t count,
        float scalar
    );

private:
    VulkanContext* context_;

    VkShaderModule shader_module_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline pipeline_;

    std::uint32_t binding_count_;

    void create_shader_module(
        const std::vector<std::uint32_t>& code
    );

    void create_descriptor_set_layout();

    void create_pipeline_layout();

    void create_compute_pipeline();

    void destroy();

    std::vector<std::uint32_t> read_spirv(
        const std::string& path
    ) const;

    void check(
        VkResult result,
        const char* operation
    ) const;
};

} // namespace minitorch
