#include "ad/ops/nodeops.hpp"
#include "ad/utils/debug.hpp"
#include "tensor.hpp"

namespace ag {
namespace detail {

std::shared_ptr<Node> gather_nodeops(const std::shared_ptr<Node>& input, const std::shared_ptr<Node>& dim, const std::shared_ptr<Node>& index) {
    int d = static_cast<int>(dim->value.to_cpu().data<float>()[0]);
    Tensor Y = OwnTensor::gather(input->value, d, index->value);
    auto n = std::make_shared<Node>(Y, Op::Gather, input->requires_grad(), "gather");
    n->inputs = {input, dim, index};
    if (input) input->child_grad_count++;
    if (dim) dim->child_grad_count++;
    if (index) index->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> scatter_add_nodeops(const std::shared_ptr<Node>& self, const std::shared_ptr<Node>& dim, const std::shared_ptr<Node>& index, const std::shared_ptr<Node>& src) {
    int d = static_cast<int>(dim->value.to_cpu().data<float>()[0]);
    Tensor Y = self->value.clone();
    OwnTensor::scatter_add(Y, d, index->value, src->value);
    auto n = std::make_shared<Node>(Y, Op::ScatterAdd, (self->requires_grad() || src->requires_grad()), "scatter_add");
    n->inputs = {self, dim, index, src};
    if (self) self->child_grad_count++;
    if (dim) dim->child_grad_count++;
    if (index) index->child_grad_count++;
    if (src) src->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

} // namespace detail
} // namespace ag
