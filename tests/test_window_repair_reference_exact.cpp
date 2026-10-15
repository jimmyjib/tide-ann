// The reference planner compiled with -ffp-contract=off (CMakeLists.txt).
#define REFERENCE_TRACE 1
#include "test_window_repair_reference.inc"

namespace repair_reference {

ReferencePlan exact_plan(const WindowRepairConfig& config, uint32_t K, uint32_t window, uint64_t queries,
                         uint64_t total_bytes, const std::vector<uint32_t>& membership,
                         const std::vector<uint64_t>& row_bytes, const std::vector<uint64_t>& work,
                         const std::vector<uint64_t>& nb, const std::vector<uint32_t>& hist,
                         const std::vector<double>& price, const std::vector<double>& previous,
                         std::vector<uint32_t> cooldown, const std::vector<uint64_t>* allowed,
                         ReferenceTrace* trace) {
    return reference_plan(config, K, window, queries, total_bytes, membership, row_bytes, work, nb, hist, price,
                          previous, std::move(cooldown), allowed, trace);
}

WindowRepairBalanceResult exact_balance(const WindowRepairConfig& config, uint32_t K,
                                        const std::vector<uint32_t>& owner_hist, const std::vector<double>& pred,
                                        const std::vector<double>& price, const std::vector<uint64_t>& row_bytes,
                                        uint64_t budget_bytes, uint32_t window, std::vector<uint32_t>& membership,
                                        std::vector<double>& W, std::vector<uint32_t>& cooldown_until,
                                        std::vector<uint8_t>& moved_now, std::vector<WindowBurstAction>& actions,
                                        const std::vector<uint64_t>* allowed, ReferenceTrace* trace) {
    if (trace) {
        double scale = 0.0;
        for (const double w : W) scale += std::fabs(w);
        trace->scale = scale > 0.0 ? scale : 1.0;
    }
    return config.balance_target == WindowRepairBalanceTarget::UNDERLOADED
        ? reference_balance_underloaded(config, K, owner_hist, pred, price, row_bytes, budget_bytes, window,
                                        membership, W, cooldown_until, moved_now, actions, allowed, trace)
        : reference_balance_lightest(config, K, owner_hist, pred, price, row_bytes, budget_bytes, window,
                                     membership, W, cooldown_until, moved_now, actions, allowed, trace);
}

}  // namespace repair_reference
