#include "minitorch/vulkan/pipeline.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace minitorch {

VulkanComputePipeline::VulkanComputePipeline(
    VulkanContext& context,
    const std::string& shader_path
)
    : context_(&context),
      shader_module_(VK_NULL_HANDLE),
      descriptor_set_layout_(VK_NULL_HANDLE),
      pipeline_layout_(VK_NULL_HANDLE),
      pipeline_(VK_NULL_HANDLE) {

    if (!context.available()) {
        throw std::runtime_error(
            "VulkanComputePipeline requires an available Vulkan context"
        );
    }

    const auto code = read_spirv(shader_path);

    create_shader_module(code);
    create_descriptor_set_layout();
    create_pipeline_layout();
    create_compute_pipeline();
}

VulkanComputePipeline::~VulkanComputePipeline() {
    destroy();
}

VulkanComputePipeline::VulkanComputePipeline(
    VulkanComputePipeline&& other
) noexcept
    : context_(other.context_),
      shader_module_(other.shader_module_),
      descriptor_set_layout_(other.descriptor_set_layout_),
      pipeline_layout_(other.pipeline_layout_),
      pipeline_(other.pipeline_) {

    other.context_ = nullptr;
    other.shader_module_ = VK_NULL_HANDLE;
    other.descriptor_set_layout_ = VK_NULL_HANDLE;
    other.pipeline_layout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
}

VulkanComputePipeline& VulkanComputePipeline::operator=(
    VulkanComputePipeline&& other
) noexcept {

    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = other.context_;
    shader_module_ = other.shader_module_;
    descriptor_set_layout_ = other.descriptor_set_layout_;
    pipeline_layout_ = other.pipeline_layout_;
    pipeline_ = other.pipeline_;

    other.context_ = nullptr;
    other.shader_module_ = VK_NULL_HANDLE;
    other.descriptor_set_layout_ = VK_NULL_HANDLE;
    other.pipeline_layout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;

    return *this;
}

bool VulkanComputePipeline::valid() const {
    return
        context_ != nullptr &&
        context_->available() &&
        shader_module_ != VK_NULL_HANDLE &&
        descriptor_set_layout_ != VK_NULL_HANDLE &&
        pipeline_layout_ != VK_NULL_HANDLE &&
        pipeline_ != VK_NULL_HANDLE;
}

std::vector<std::uint32_t>
VulkanComputePipeline::read_spirv(
    const std::string& path
) const {

    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to open SPIR-V shader: " + path
        );
    }

    const std::streamsize size =
        file.tellg();

    if (size <= 0) {
        throw std::runtime_error(
            "SPIR-V shader is empty: " + path
        );
    }

    if (
        size % static_cast<std::streamsize>(
            sizeof(std::uint32_t)
        ) != 0
    ) {
        throw std::runtime_error(
            "SPIR-V shader size is not aligned: " + path
        );
    }

    file.seekg(0);

    std::vector<std::uint32_t> code(
        static_cast<std::size_t>(size)
        / sizeof(std::uint32_t)
    );

    file.read(
        reinterpret_cast<char*>(code.data()),
        size
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read SPIR-V shader: " + path
        );
    }

    return code;
}

void VulkanComputePipeline::create_shader_module(
    const std::vector<std::uint32_t>& code
) {

    VkShaderModuleCreateInfo info{};

    info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    info.codeSize =
        code.size() * sizeof(std::uint32_t);

    info.pCode =
        code.data();

    check(
        vkCreateShaderModule(
            context_->device(),
            &info,
            nullptr,
            &shader_module_
        ),
        "vkCreateShaderModule"
    );
}

void VulkanComputePipeline::create_descriptor_set_layout() {

    VkDescriptorSetLayoutBinding bindings[2]{};

    bindings[0].binding = 0;
    bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo info{};

    info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

    info.bindingCount = 2;
    info.pBindings = bindings;

    check(
        vkCreateDescriptorSetLayout(
            context_->device(),
            &info,
            nullptr,
            &descriptor_set_layout_
        ),
        "vkCreateDescriptorSetLayout"
    );
}

void VulkanComputePipeline::create_pipeline_layout() {

    VkPushConstantRange push_constant{};

    push_constant.stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    push_constant.offset = 0;
    push_constant.size = sizeof(std::uint32_t);

    VkPipelineLayoutCreateInfo info{};

    info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    info.setLayoutCount = 1;
    info.pSetLayouts =
        &descriptor_set_layout_;

    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges =
        &push_constant;

    check(
        vkCreatePipelineLayout(
            context_->device(),
            &info,
            nullptr,
            &pipeline_layout_
        ),
        "vkCreatePipelineLayout"
    );
}

void VulkanComputePipeline::create_compute_pipeline() {

    VkPipelineShaderStageCreateInfo shader_stage{};

    shader_stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    shader_stage.stage =
        VK_SHADER_STAGE_COMPUTE_BIT;

    shader_stage.module =
        shader_module_;

    shader_stage.pName =
        "main";

    VkComputePipelineCreateInfo info{};

    info.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;

    info.stage =
        shader_stage;

    info.layout =
        pipeline_layout_;

    check(
        vkCreateComputePipelines(
            context_->device(),
            VK_NULL_HANDLE,
            1,
            &info,
            nullptr,
            &pipeline_
        ),
        "vkCreateComputePipelines"
    );
}

void VulkanComputePipeline::dispatch(
    VulkanCompute& compute,
    const VulkanBuffer& input,
    VulkanBuffer& output,
    std::size_t count
) {

    if (!valid()) {
        throw std::runtime_error(
            "Invalid VulkanComputePipeline"
        );
    }

    if (!compute.valid()) {
        throw std::runtime_error(
            "Invalid VulkanCompute"
        );
    }

    if (!input.valid()) {
        throw std::runtime_error(
            "Invalid Vulkan input buffer"
        );
    }

    if (!output.valid()) {
        throw std::runtime_error(
            "Invalid Vulkan output buffer"
        );
    }

    if (count == 0) {
        throw std::invalid_argument(
            "Vulkan compute count must be greater than zero"
        );
    }

    const std::size_t required_size =
        count * sizeof(float);

    if (input.size() < required_size) {
        throw std::invalid_argument(
            "Input buffer is too small"
        );
    }

    if (output.size() < required_size) {
        throw std::invalid_argument(
            "Output buffer is too small"
        );
    }

    VkDescriptorPoolSize pool_size{};

    pool_size.type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    pool_size.descriptorCount = 2;

    VkDescriptorPoolCreateInfo pool_info{};

    pool_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;

    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool descriptor_pool =
        VK_NULL_HANDLE;

    check(
        vkCreateDescriptorPool(
            context_->device(),
            &pool_info,
            nullptr,
            &descriptor_pool
        ),
        "vkCreateDescriptorPool"
    );

    try {

        VkDescriptorSetAllocateInfo allocate_info{};

        allocate_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;

        allocate_info.descriptorPool =
            descriptor_pool;

        allocate_info.descriptorSetCount = 1;

        allocate_info.pSetLayouts =
            &descriptor_set_layout_;

        VkDescriptorSet descriptor_set =
            VK_NULL_HANDLE;

        check(
            vkAllocateDescriptorSets(
                context_->device(),
                &allocate_info,
                &descriptor_set
            ),
            "vkAllocateDescriptorSets"
        );

        VkDescriptorBufferInfo input_info{};

        input_info.buffer =
            input.handle();

        input_info.offset = 0;
        input_info.range = required_size;

        VkDescriptorBufferInfo output_info{};

        output_info.buffer =
            output.handle();

        output_info.offset = 0;
        output_info.range = required_size;

        VkWriteDescriptorSet writes[2]{};

        writes[0].sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

        writes[0].dstSet =
            descriptor_set;

        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;

        writes[0].descriptorCount = 1;

        writes[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        writes[0].pBufferInfo =
            &input_info;

        writes[1].sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

        writes[1].dstSet =
            descriptor_set;

        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;

        writes[1].descriptorCount = 1;

        writes[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        writes[1].pBufferInfo =
            &output_info;

        vkUpdateDescriptorSets(
            context_->device(),
            2,
            writes,
            0,
            nullptr
        );

        compute.begin();

        VkCommandBuffer command =
            compute.command_buffer();

        vkCmdBindPipeline(
            command,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_
        );

        vkCmdBindDescriptorSets(
            command,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_,
            0,
            1,
            &descriptor_set,
            0,
            nullptr
        );

        const std::uint32_t shader_count =
            static_cast<std::uint32_t>(count);

        vkCmdPushConstants(
            command,
            pipeline_layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(std::uint32_t),
            &shader_count
        );

        const std::uint32_t groups =
            static_cast<std::uint32_t>(
                (count + 63) / 64
            );

        vkCmdDispatch(
            command,
            groups,
            1,
            1
        );

        compute.end();

        compute.submit_and_wait();

        vkDestroyDescriptorPool(
            context_->device(),
            descriptor_pool,
            nullptr
        );

    }
    catch (...) {

        vkDestroyDescriptorPool(
            context_->device(),
            descriptor_pool,
            nullptr
        );

        throw;
    }
}

void VulkanComputePipeline::destroy() {

    if (context_ == nullptr) {
        return;
    }

    VkDevice device =
        context_->device();

    if (device == VK_NULL_HANDLE) {
        context_ = nullptr;
        return;
    }

    if (pipeline_ != VK_NULL_HANDLE) {

        vkDestroyPipeline(
            device,
            pipeline_,
            nullptr
        );

        pipeline_ =
            VK_NULL_HANDLE;
    }

    if (pipeline_layout_ != VK_NULL_HANDLE) {

        vkDestroyPipelineLayout(
            device,
            pipeline_layout_,
            nullptr
        );

        pipeline_layout_ =
            VK_NULL_HANDLE;
    }

    if (
        descriptor_set_layout_
        != VK_NULL_HANDLE
    ) {

        vkDestroyDescriptorSetLayout(
            device,
            descriptor_set_layout_,
            nullptr
        );

        descriptor_set_layout_ =
            VK_NULL_HANDLE;
    }

    if (shader_module_ != VK_NULL_HANDLE) {

        vkDestroyShaderModule(
            device,
            shader_module_,
            nullptr
        );

        shader_module_ =
            VK_NULL_HANDLE;
    }

    context_ = nullptr;
}

void VulkanComputePipeline::check(
    VkResult result,
    const char* operation
) const {

    if (result != VK_SUCCESS) {

        throw std::runtime_error(
            std::string(operation) +
            " failed with VkResult " +
            std::to_string(
                static_cast<int>(result)
            )
        );
    }
}

} // namespace minitorch
