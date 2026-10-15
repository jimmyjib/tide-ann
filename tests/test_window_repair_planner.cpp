// Hand-computed expectations for the local-loop repair planner.
// Every number below is derived in the comments from the fixture counters;
// nothing is read back from the implementation. The randomised tests compare
// the production-compiled planner with a reference written from the rule
// definitions (calibrate, refine, both balance targets) for K in {2, 3, 4, 8,
// 16}, compiled without floating-point contraction in its own source file
// (test_window_repair_reference.h: decisions exact, rounding boundaries shown and
// bounded); the owner limits (K = 1 and 65 refused, K = 64 with mask bit 63) and
// the K = 8 balance target are checked by hand.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <random>
#include <string>
#include <vector>

#include "coordinator/window_repair_planner.h"
#include "test_window_repair_reference.h"

namespace {
using namespace tide;
void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", why); std::abort(); }
}
bool near(double a, double b, double tol = 1e-9) { return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol; }
using Row = std::array<uint32_t, 4>;
std::vector<uint32_t> hist_of(const std::vector<Row>& rows) {
    std::vector<uint32_t> out;
    for (const auto& row : rows) out.insert(out.end(), row.begin(), row.end());
    return out;
}
std::vector<uint32_t> owners_of(const std::vector<WindowBurstAction>& actions) {
    std::vector<uint32_t> out;
    for (const auto& a : actions) out.push_back(a.node);
    return out;
}

// Fixture A: 16 nodes, owner p = v / 4, prices {1, 1, 0.5, 1.5}, 100-byte rows.
//   w  = {10,0,20,0, 10,10,0,0, 0,0,20,40, 20,0,10,0}
//   nb = { 0,10,0,10, 0,0,10,10, 10,10,0,0, 0,10,0,0}
//   Q = 10  ->  pred = (w + nb) / 10 = {1,1,2,1, 1,1,1,1, 1,1,2,4, 2,1,1,0}
//   owner sums of pred: {5, 4, 8, 4};  W = price * sums = {5, 4, 4, 6}
//   mean(W) = 4.75; skew = 6 / 4.75 = 24/19
struct FixtureA {
    std::vector<uint32_t> membership = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3};
    std::vector<uint64_t> work = {10,0,20,0, 10,10,0,0, 0,0,20,40, 20,0,10,0};
    std::vector<uint64_t> nb_work = {0,10,0,10, 0,0,10,10, 10,10,0,0, 0,10,0,0};
    std::vector<double> price = {1.0, 1.0, 0.5, 1.5};
    std::vector<uint64_t> row_bytes = std::vector<uint64_t>(16, 100);
    std::vector<double> pred = {1,1,2,1, 1,1,1,1, 1,1,2,4, 2,1,1,0};
    std::vector<double> W = {5, 4, 4, 6};
    // Histogram rows chosen so exactly nodes 2,3,5,7,9,11,12,13 have gain > 0:
    //   v11 {5,0,1,0}: own 2 -> q 0, gain 4      v12 {0,0,4,1}: own 3 -> q 2, gain 3
    //   v2  {1,3,0,0}: own 0 -> q 1, gain 2      v5  {0,1,3,0}: own 1 -> q 2, gain 2
    //   v13 {0,2,0,0}: own 3 -> q 1, gain 2      v3  {1,2,1,2}: own 0 -> q 1 (tie 1 vs 3, lowest index), gain 1
    //   v7  {0,0,0,1}: own 1 -> q 3, gain 1      v9  {0,0,0,1}: own 2 -> q 3, gain 1
    // Every other row has gain <= 0 (v6 {1,1,0,0} and v10 {1,0,1,0} tie at 0, v15 is empty).
    std::vector<uint32_t> hist = hist_of({
        {3,0,0,0}, {2,1,0,0}, {1,3,0,0}, {1,2,1,2},
        {0,3,0,0}, {0,1,3,0}, {1,1,0,0}, {0,0,0,1},
        {0,0,2,0}, {0,0,0,1}, {1,0,1,0}, {5,0,1,0},
        {0,0,4,1}, {0,2,0,0}, {0,0,0,2}, {0,0,0,0}});
    WindowRepairConfig config() const {
        WindowRepairConfig c; c.enabled = true; c.eps = 0.1; c.cooldown_windows = 2;
        c.tau = 0.2; c.skew_trigger = 1.05; c.weight = WindowRepairWeight::WNB; return c;
    }
};

void test_calibration() {
    FixtureA f;
    WindowRepairConfig config = f.config();
    std::vector<double> pred;
    std::vector<double> W{};
    WindowRepairCalibration out; std::string error;
    check(window_repair_calibrate(config, 4, f.work, f.nb_work, 10, f.membership, f.price, pred, W, &out, &error),
          "first calibration");
    check(pred == f.pred, "first window pred is (w + nb_work) / Q");
    check(near(W[0], 5) && near(W[1], 4) && near(W[2], 4) && near(W[3], 6), "W = price * owner sums of pred");
    check(near(out.skew, 24.0 / 19.0) && out.queries == 10 && out.active_nodes == 8, "skew, Q and active (w > 0) nodes");
    // measured_share = owner sums of w / sum(w) = {30, 20, 60, 30} / 140
    check(near(out.measured_share[0], 30.0 / 140) && near(out.measured_share[1], 20.0 / 140) &&
          near(out.measured_share[2], 60.0 / 140) && near(out.measured_share[3], 30.0 / 140), "measured share per owner");
    // Second window: Q = 20 with w2 = 4w, nb2 = 4nb -> nb_q = 2 * pred; ema .5 -> pred2 = 1.5 * pred
    std::vector<uint64_t> w2, nb2;
    for (size_t v = 0; v < 16; ++v) { w2.push_back(4 * f.work[v]); nb2.push_back(4 * f.nb_work[v]); }
    check(window_repair_calibrate(config, 4, w2, nb2, 20, f.membership, f.price, pred, W, &out, &error), "second calibration");
    for (size_t v = 0; v < 16; ++v) check(near(pred[v], 1.5 * f.pred[v]), "EMA on the second window with per-query normalisation");
    check(near(W[0], 7.5) && near(W[1], 6) && near(W[2], 6) && near(W[3], 9) && near(out.skew, 24.0 / 19.0),
          "second-window W scales with the smoothed weights");
    // ema .25 from the first-window pred: .25 * 2 + .75 * 1 = 1.25
    config.ema = 0.25; pred = f.pred;
    check(window_repair_calibrate(config, 4, w2, nb2, 20, f.membership, f.price, pred, W, &out, &error) && near(pred[11], 5.0) &&
          near(pred[2], 2.5), "ema weight is applied to the new observation");
    // Calibration case: the measured work replaces the in-degree prior.
    // graph 0->{1,2}, 1->{0,3}, 2->{0,3}, 3->{1,2}; work {10,0,4,0}; A^T work = {4,10,10,4}
    std::vector<double> pred4; std::vector<double> W4{};
    check(window_repair_calibrate(config, 4, {10, 0, 4, 0}, {4, 10, 10, 4}, 1, {0, 0, 1, 1}, {1, 1, 1, 1}, pred4, W4, &out, &error),
          "calibration case");
    check(pred4 == std::vector<double>({14, 10, 14, 4}) && near(W4[0], 24) && near(W4[1], 18) && near(W4[2], 0) && near(W4[3], 0),
          "pred = work + in-neighbour work; W = {24, 18, 0, 0}");
    check(near(out.measured_share[0], 10.0 / 14) && near(out.measured_share[1], 4.0 / 14) && out.active_nodes == 2,
          "calibration case: measured share and active nodes");
    // Domain: zero queries, bad shapes and a non-positive price fail closed.
    std::vector<double> untouched = pred4;
    check(!window_repair_calibrate(config, 4, {10, 0, 4, 0}, {4, 10, 10, 4}, 0, {0, 0, 1, 1}, {1, 1, 1, 1}, pred4, W4, &out, &error) &&
          !window_repair_calibrate(config, 4, {10, 0, 4}, {4, 10, 10, 4}, 1, {0, 0, 1, 1}, {1, 1, 1, 1}, pred4, W4, &out, &error) &&
          !window_repair_calibrate(config, 4, {10, 0, 4, 0}, {4, 10, 10, 4}, 1, {0, 0, 1, 1}, {1, 0, 1, 1}, pred4, W4, &out, &error) &&
          pred4 == untouched, "invalid calibration input leaves pred untouched");
    std::printf("test_calibration OK\n");
}

// Weight W (the default): pred = w / Q, nb_work empty. Fixture A's counters:
//   pred = w / 10 = {1,0,2,0, 1,1,0,0, 0,0,2,4, 2,0,1,0}; owner sums {3, 2, 6, 3}
//   W = price * sums = {3, 2, 3, 4.5}; mean 3.125, skew 4.5 / 3.125 = 1.44
void test_weight_w() {
    FixtureA f;
    WindowRepairConfig w_config = f.config(); w_config.weight = WindowRepairWeight::W;
    check(WindowRepairConfig{}.weight == WindowRepairWeight::W, "W is the default weight");
    const std::vector<uint64_t> none;
    std::vector<double> pred, W;
    WindowRepairCalibration out; std::string error;
    check(window_repair_calibrate(w_config, 4, f.work, none, 10, f.membership, f.price, pred, W, &out, &error),
          "weight W calibrates without nbwork");
    check(pred == std::vector<double>({1,0,2,0, 1,1,0,0, 0,0,2,4, 2,0,1,0}), "weight W: pred = w / Q");
    check(W == std::vector<double>({3, 2, 3, 4.5}) && out.skew == 4.5 / 3.125 && out.active_nodes == 8 &&
          near(out.measured_share[2], 60.0 / 140), "weight W: W = price * owner sums of w / Q");
    // Shapes: W takes no nbwork, WNB needs one entry per node; pred is left untouched on refusal.
    const std::vector<double> kept = pred;
    check(!window_repair_calibrate(w_config, 4, f.work, f.nb_work, 10, f.membership, f.price, pred, W, &out, &error) &&
          error.find("shape") != std::string::npos && pred == kept, "weight W refuses an nbwork section");
    WindowRepairConfig wnb = f.config();
    check(!window_repair_calibrate(wnb, 4, f.work, none, 10, f.membership, f.price, pred, W, &out, &error) &&
          pred == kept, "weight WNB refuses a missing nbwork section");
    WindowRepairConfig bad = w_config; bad.weight = static_cast<WindowRepairWeight>(7);
    check(!window_repair_calibrate(bad, 4, f.work, none, 10, f.membership, f.price, pred, W, &out, &error) &&
          error.find("weight") != std::string::npos, "an unknown weight value fails closed");
    // The same window under W and WNB: only the nbwork term differs (pred_wnb - pred_w = nb / Q).
    std::vector<double> pred_wnb, W_wnb;
    check(window_repair_calibrate(wnb, 4, f.work, f.nb_work, 10, f.membership, f.price, pred_wnb, W_wnb, &out, &error) &&
          pred_wnb == f.pred && W_wnb == f.W, "weight WNB: Fixture A's (w + nb) / Q");
    for (size_t v = 0; v < 16; ++v)
        check(pred_wnb[v] - pred[v] == double(f.nb_work[v]) / 10.0, "W and WNB differ by nb / Q only");
    // Whole-window plan under W: the refine/balance rules see the W-calibrated loads.
    const auto plan = window_repair_plan(w_config, 4, WindowRepairPhase::IDLE, 1, 10, 1600, f.membership, f.row_bytes,
                                         f.work, none, f.hist, f.price, {}, {});
    check(plan.valid && plan.planned && plan.pred == pred && plan.calibration.W == std::vector<double>({3, 2, 3, 4.5}),
          "a weight-W plan calibrates from w alone");
    check(!window_repair_plan(w_config, 4, WindowRepairPhase::IDLE, 1, 10, 1600, f.membership, f.row_bytes, f.work,
                              f.nb_work, f.hist, f.price, {}, {}).valid &&
          !window_repair_plan(wnb, 4, WindowRepairPhase::IDLE, 1, 10, 1600, f.membership, f.row_bytes, f.work, none,
                              f.hist, f.price, {}, {}).valid, "a plan whose nbwork does not match the weight is refused");
    // Sections: the owner histogram always, nbwork under WNB only.
    check(window_repair_structure_sections(w_config) == kWindowStructureOwnerHist &&
          window_repair_structure_sections(wnb) == (kWindowStructureOwnerHist | kWindowStructureNbWork),
          "structure sections follow the weight");
    std::printf("test_weight_w OK\n");
}

void test_refine() {
    FixtureA f;
    const WindowRepairConfig config = f.config();  // eps .1 -> limit = 1.1 * 4.75 = 5.225
    auto run = [&](uint64_t budget, std::vector<uint32_t> cooldown, std::vector<uint8_t> moved_now,
                   std::vector<uint32_t>& membership, std::vector<double>& W, std::vector<WindowBurstAction>& actions) {
        membership = f.membership; W = f.W; actions.clear();
        return window_repair_refine(config, 4, f.hist, f.pred, f.price, f.row_bytes, budget, 1, membership, W, cooldown, moved_now, actions);
    };
    std::vector<uint32_t> membership; std::vector<double> W{}; std::vector<WindowBurstAction> actions;
    // Walk order (gain desc, pred desc, id asc): 11, 12, 2, 5, 13, 3, 7, 9.
    //   v11 -> 0: 5 + 1*4 = 9 > 5.225 rejected      v12 -> 2: 4 + .5*2 = 5 accepted (W {5,4,5,3})
    //   v2  -> 1: 4 + 1*2 = 6 rejected              v5  -> 2: 5 + .5 = 5.5 rejected
    //   v13 -> 1: 4 + 1 = 5 accepted (W {5,5,5,1.5}) v3 -> 1: 5 + 1 = 6 rejected
    //   v7  -> 3: 1.5 + 1.5 = 3 accepted (W {5,4,5,3}) v9 -> 3: 3 + 1.5 = 4.5 accepted (W {5,4,4.5,4.5})
    std::vector<uint32_t> cooldown(16, 0); std::vector<uint8_t> moved_now(16, 0);
    auto full = run(1600, cooldown, moved_now, membership, W, actions);
    check(full.candidates == 8 && full.moved == 4 && full.rejected_balance == 4 && full.bytes == 400, "full-budget refine counts");
    check(owners_of(actions) == std::vector<uint32_t>({12, 13, 7, 9}), "accepted moves in walk order");
    check(actions[0].source_owner == 3 && actions[0].target_owner == 2 && actions[1].target_owner == 1 &&
          actions[2].source_owner == 1 && actions[2].target_owner == 3 && actions[3].source_owner == 2 && actions[3].target_owner == 3,
          "move directions");
    check(actions[0].adjacency_bytes == 100 && actions[3].adjacency_bytes == 100, "actions carry the row bytes");
    check(near(W[0], 5) && near(W[1], 4) && near(W[2], 4.5) && near(W[3], 4.5), "W after the four moves");
    check(membership[12] == 2 && membership[13] == 1 && membership[7] == 3 && membership[9] == 3 && membership[11] == 2,
          "local membership copy updated only for accepted moves");
    // cooldown_until = window + cooldown (1 + 2) for moved nodes; moved_now marked
    {
        std::vector<uint32_t> cd(16, 0); std::vector<uint8_t> mn(16, 0);
        std::vector<uint32_t> m = f.membership; std::vector<double> w = f.W; std::vector<WindowBurstAction> a;
        window_repair_refine(config, 4, f.hist, f.pred, f.price, f.row_bytes, 1600, 1, m, w, cd, mn, a);
        check(cd[12] == 3 && cd[13] == 3 && cd[7] == 3 && cd[9] == 3 && cd[11] == 0 && mn[12] && mn[9] && !mn[11],
              "cooldown and moved_now mark accepted moves only");
    }
    // Byte budget 250: v12 (100) and v13 (200) fit; v3 would need 300 -> the walk stops there.
    auto capped = run(250, cooldown, moved_now, membership, W, actions);
    check(capped.candidates == 8 && capped.moved == 2 && capped.rejected_balance == 3 && capped.bytes == 200 &&
          owners_of(actions) == std::vector<uint32_t>({12, 13}), "byte budget stops the walk before v3");
    check(near(W[0], 5) && near(W[1], 5) && near(W[2], 5) && near(W[3], 1.5), "W after the capped walk");
    // Cooldown exclusion: v11 on cooldown until window 2 is not a candidate.
    std::vector<uint32_t> on_cooldown(16, 0); on_cooldown[11] = 2;
    auto excluded = run(1600, on_cooldown, moved_now, membership, W, actions);
    check(excluded.candidates == 7 && excluded.rejected_balance == 3 && excluded.moved == 4, "cooldown removes v11 from candidacy");
    // Already moved this window: v12 excluded -> v5 now fits on owner 2 (4 + .5), v13 on 1 (3 + 1), v3 on 1 (4 + 1 = 5).
    std::vector<uint8_t> moved12(16, 0); moved12[12] = 1;
    auto single = run(1600, cooldown, moved12, membership, W, actions);
    check(single.candidates == 7 && single.moved == 3 && single.rejected_balance == 4 &&
          owners_of(actions) == std::vector<uint32_t>({5, 13, 3}), "a node moved earlier this window is never a candidate again");
    check(near(W[0], 4) && near(W[1], 5) && near(W[2], 4.5) && near(W[3], 4.5), "W after the alternative walk");
    // eps 0: limit = mean(W) = 4.75. v11 (9), v12 (5), v2 (6) break it; v5 -> 2 fits (4 + .5 = 4.5,
    // W {5,3,4.5,6}); v13 -> 1 fits (3 + 1 = 4, W {5,4,4.5,4.5}); v3 (5), v7 (6), v9 (6) break it.
    WindowRepairConfig tight = config; tight.eps = 0.0;
    std::vector<uint32_t> m = f.membership; std::vector<double> w = f.W; std::vector<WindowBurstAction> a;
    std::vector<uint32_t> cd(16, 0); std::vector<uint8_t> mn(16, 0);
    auto strict = window_repair_refine(tight, 4, f.hist, f.pred, f.price, f.row_bytes, 1600, 1, m, w, cd, mn, a);
    check(strict.candidates == 8 && strict.moved == 2 && strict.rejected_balance == 6 && strict.bytes == 200 &&
          owners_of(a) == std::vector<uint32_t>({5, 13}) && m[5] == 2 && m[13] == 1 && m[12] == 3 &&
          near(w[0], 5) && near(w[1], 4) && near(w[2], 4.5) && near(w[3], 4.5),
          "eps 0 admits only the two moves that keep the target at or below the mean");
    std::printf("test_refine OK\n");
}

// Balance case: only edge-admissible heavy nodes move.
// rows 0->{1,2}, 1->{3,4,5}, 2->{6,7,8}, 3..5->{1}, 6..8->{2}; weights = in-degree
// = {0,4,4,1,1,1,1,1,1}; 10-byte rows.
void test_balance() {
    const std::vector<double> pred = {0, 4, 4, 1, 1, 1, 1, 1, 1};
    const std::vector<double> ones = {1, 1, 1, 1};
    const std::vector<uint64_t> rows(9, 10);
    WindowRepairConfig config; config.enabled = true; config.tau = 0.5; config.skew_trigger = 1.05; config.cooldown_windows = 1;
    // membership {0,0,0,1,1,1,0,0,0}: W = {11, 3, 0, 0}; h: 0 {2,0,0,0}, 1 {0,3,0,0}, 2 {3,0,0,0}, 3..8 {1,0,0,0}
    {
        std::vector<uint32_t> m = {0,0,0,1,1,1,0,0,0};
        const auto hist = hist_of({{2,0,0,0}, {0,3,0,0}, {3,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}});
        std::vector<double> W = {11, 3, 0, 0};
        std::vector<uint32_t> cd(9, 0); std::vector<uint8_t> mn(9, 0); std::vector<WindowBurstAction> a;
        auto out = window_repair_balance(config, 4, hist, pred, ones, rows, 1000, 1, m, W, cd, mn, a);
        check(out.triggered && near(out.skew_before, 11 / 3.5) && out.moved == 0 && a.empty(),
              "lightest owner 2 admits no node at tau .5 (moved 0)");
    }
    // membership {0,0,0,1,1,1,1,1,1}: W = {8, 6, 0, 0}; h: 2 {0,3,0,0}, 6..8 {1,0,0,0}
    const auto hist2 = hist_of({{2,0,0,0}, {0,3,0,0}, {0,3,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}});
    const std::vector<uint32_t> membership2 = {0,0,0,1,1,1,1,1,1};
    {
        std::vector<uint32_t> m = membership2; std::vector<double> W = {8, 6, 0, 0};
        std::vector<uint32_t> cd(9, 0); std::vector<uint8_t> mn(9, 0); std::vector<WindowBurstAction> a;
        auto out = window_repair_balance(config, 4, hist2, pred, ones, rows, 1000, 1, m, W, cd, mn, a);
        check(out.triggered && out.moved == 0, "tau .5 still admits nothing toward the empty owner (moved 0)");
    }
    // tau 0: step 1 peak 0 (8) low 2 (0) gap 8: v1 (4 <= 4) moves -> {4,6,4,0};
    //        step 2 peak 1 (6) low 3 (0) gap 6: v3 (1 <= 3) -> {4,5,4,1}; step 3 gap 4: v4 -> {4,4,4,2};
    //        step 4 peak 0 (4, first max) low 3 gap 2: owner-0 list [v1 used, v2 weight 4 >= 2] exhausted -> stop.
    WindowRepairConfig open = config; open.tau = 0.0;
    {
        std::vector<uint32_t> m = membership2; std::vector<double> W = {8, 6, 0, 0};
        std::vector<uint32_t> cd(9, 0); std::vector<uint8_t> mn(9, 0); std::vector<WindowBurstAction> a;
        auto out = window_repair_balance(open, 4, hist2, pred, ones, rows, 1000, 2, m, W, cd, mn, a);
        check(out.triggered && out.moved == 3 && out.bytes == 30 && owners_of(a) == std::vector<uint32_t>({1, 3, 4}),
              "tau 0 admits the hub then two leaves (moved >= 1; exact sequence 1, 3, 4)");
        check(a[0].source_owner == 0 && a[0].target_owner == 2 && a[1].source_owner == 1 && a[1].target_owner == 3 &&
              a[2].target_owner == 3, "balance action directions");
        check(near(W[0], 4) && near(W[1], 4) && near(W[2], 4) && near(W[3], 2) && m[1] == 2 && m[3] == 3 && m[4] == 3 &&
              cd[1] == 3 && cd[3] == 3 && mn[4] && !mn[2], "sums, membership, cooldown (window 2 + 1) and moved_now");
    }
    // Price-scaled sums {1,1,2,1}: v1 leaves owner 0 at weight 4 and lands on owner 2 at 2 * 4 = 8,
    // so owner 2 becomes the peak (8) with an empty candidate list: the greedy stops after one move.
    {
        std::vector<uint32_t> m = membership2; std::vector<double> W = {8, 6, 0, 0};
        std::vector<uint32_t> cd(9, 0); std::vector<uint8_t> mn(9, 0); std::vector<WindowBurstAction> a;
        auto out = window_repair_balance(open, 4, hist2, pred, {1, 1, 2, 1}, rows, 1000, 1, m, W, cd, mn, a);
        check(out.moved == 1 && owners_of(a) == std::vector<uint32_t>({1}) && near(W[0], 4) && near(W[1], 6) && near(W[2], 8) && near(W[3], 0),
              "a moved node is re-priced with its destination price");
    }
    // Budget 15 with 10-byte rows: the second move reaches the budget checkpoint (one check after each move) and
    // ends the greedy.
    {
        std::vector<uint32_t> m = membership2; std::vector<double> W = {8, 6, 0, 0};
        std::vector<uint32_t> cd(9, 0); std::vector<uint8_t> mn(9, 0); std::vector<WindowBurstAction> a;
        auto out = window_repair_balance(open, 4, hist2, pred, ones, rows, 15, 1, m, W, cd, mn, a);
        check(out.moved == 2 && out.bytes == 20 && owners_of(a) == std::vector<uint32_t>({1, 3}) && near(W[1], 5) && near(W[3], 1),
              "byte budget exhaustion after the crossing move");
        std::vector<uint32_t> mv0 = membership2; std::vector<double> W0 = {8, 6, 0, 0}; std::vector<WindowBurstAction> a0;
        auto zero = window_repair_balance(open, 4, hist2, pred, ones, rows, 0, 1, mv0, W0, cd, mn, a0);
        check(!zero.triggered && zero.moved == 0 && a0.empty(), "zero budget never triggers");
        std::vector<double> flat = {5, 5, 5, 5}; std::vector<WindowBurstAction> a1;
        auto quiet = window_repair_balance(open, 4, hist2, pred, ones, rows, 1000, 1, mv0, flat, cd, mn, a1);
        check(!quiet.triggered && near(quiet.skew_before, 1.0) && a1.empty(), "skew at or below the trigger never triggers");
    }
    // Improving fallback: owner 0 holds v0 (6, admissible toward 1) and v1 (3, not admissible); gap 9,
    // half-gap 4.5: no admissible node at or below it, so the heaviest improving node v0 moves.
    {
        const auto hist = hist_of({{0,1,0,0}, {1,0,0,0}});
        std::vector<uint32_t> m = {0, 0}; std::vector<double> W = {9, 0, 0, 0};
        std::vector<uint32_t> cd(2, 0); std::vector<uint8_t> mn(2, 0); std::vector<WindowBurstAction> a;
        auto out = window_repair_balance(config, 4, hist, {6, 3}, ones, {10, 10}, 1000, 1, m, W, cd, mn, a);
        check(out.moved == 1 && owners_of(a) == std::vector<uint32_t>({0}) && a[0].target_owner == 1 && near(W[0], 3) && near(W[1], 6),
              "largest improving move when nothing fits half the gap");
        // Cooldown and moved-this-window rows are never admissible even at tau 0.
        std::vector<uint32_t> mv2 = {0, 0}; std::vector<double> W2 = {9, 0, 0, 0};
        std::vector<uint32_t> cd2 = {2, 0}; std::vector<uint8_t> mn2(2, 0); std::vector<WindowBurstAction> a2;
        check(window_repair_balance(open, 4, hist, {6, 3}, ones, {10, 10}, 1000, 1, mv2, W2, cd2, mn2, a2).moved == 1 &&
              owners_of(a2) == std::vector<uint32_t>({1}), "cooldown excludes v0; v1 (3 <= 4.5) moves instead");
        std::vector<uint32_t> mv3 = {0, 0}; std::vector<double> W3 = {9, 0, 0, 0};
        std::vector<uint32_t> cd3(2, 0); std::vector<uint8_t> mn3 = {1, 1}; std::vector<WindowBurstAction> a3;
        check(window_repair_balance(open, 4, hist, {6, 3}, ones, {10, 10}, 1000, 1, mv3, W3, cd3, mn3, a3).moved == 0,
              "nodes moved earlier this window are excluded");
        // deg 0 rows are never admissible.
        std::vector<uint32_t> mv4 = {0, 0}; std::vector<double> W4 = {9, 0, 0, 0};
        std::vector<uint32_t> cd4(2, 0); std::vector<uint8_t> mn4(2, 0); std::vector<WindowBurstAction> a4;
        check(window_repair_balance(open, 4, hist_of({{0,0,0,0}, {0,0,0,0}}), {6, 3}, ones, {10, 10}, 1000, 1, mv4, W4, cd4, mn4, a4).moved == 0,
              "a node without neighbours is never moved by balance");
    }
    std::printf("test_balance OK\n");
}

// Fixture C: 8 nodes, membership {0,0,0,0,1,1,2,3}, w = pred = {5,4,1,1,1,1,0,0}
// (Q 1, no nbwork), unit prices, 10-byte rows (80 total).
//   W = {11, 2, 0, 0}, mean 3.25, skew 11/3.25
//   h: 0 {3,0,0,0}, 1 {0,0,3,0} (gain 3 -> 2), 2 {2,0,0,0}, 3 {1,0,0,1} (gain 0),
//      4 {0,2,0,0}, 5 {0,1,1,0} (gain 0), 6 {0,0,1,0}, 7 {0,0,0,1}
void test_plan_union_and_phases() {
    const std::vector<uint32_t> membership = {0,0,0,0,1,1,2,3};
    const std::vector<uint64_t> work = {5,4,1,1,1,1,0,0}, nb(8, 0), rows(8, 10);
    const auto hist = hist_of({{3,0,0,0}, {0,0,3,0}, {2,0,0,0}, {1,0,0,1}, {0,2,0,0}, {0,1,1,0}, {0,0,1,0}, {0,0,0,1}});
    const std::vector<double> ones = {1, 1, 1, 1};
    WindowRepairConfig config; config.enabled = true; config.refine_budget = 0.25; config.balance_budget = 0.25;
    config.eps = 1.0; config.tau = 0.2; config.cooldown_windows = 2; config.skew_trigger = 1.05;
    config.weight = WindowRepairWeight::WNB;  // nb is passed (all zero)
    // refine (budget 20, limit 6.5): v1 -> 2 accepted (W {7,2,4,0}); balance (budget 20): skew 7/3.25 > 1.05,
    // peak 0 low 3 gap 7: v0 (h toward 3: 0 < .6) and v2 (0) inadmissible, v3 (1 >= .4) admissible and 1 <= 3.5
    // -> v3 moves ({6,2,4,1}); next peak 0 low 1 gap 4: v0 (5 >= 4) dropped, v2 not admissible toward 1 -> stop.
    auto plan = window_repair_plan(config, 4, WindowRepairPhase::IDLE, 1, 1, 80, membership, rows, work, nb, hist, ones, {}, {});
    check(plan.valid && plan.planned, "union plan runs refine and balance from IDLE");
    check(near(plan.calibration.skew, 11 / 3.25) && plan.calibration.active_nodes == 6 &&
          near(plan.calibration.W[0], 11) && near(plan.calibration.W[1], 2), "calibration precedes the moves");
    check(plan.actions.size() == 2 && plan.actions[0].node == 1 && plan.actions[0].source_owner == 0 && plan.actions[0].target_owner == 2 &&
          plan.actions[1].node == 3 && plan.actions[1].source_owner == 0 && plan.actions[1].target_owner == 3,
          "burst = refine rows then balance rows, single MOVE rows");
    check(plan.stats.refine_candidates == 1 && plan.stats.refine_moved == 1 && plan.stats.refine_bytes == 10 && plan.stats.rejected_balance == 0 &&
          plan.stats.balance_triggered && plan.stats.balance_moved == 1 && plan.stats.balance_bytes == 10 &&
          plan.stats.refine_budget_bytes == 20 && plan.stats.balance_budget_bytes == 20, "burst stats");
    check(near(plan.stats.skew_before, 11 / 3.25) && near(plan.stats.skew_after, 6 / 3.25) &&
          near(plan.W[0], 6) && near(plan.W[1], 2) && near(plan.W[2], 4) && near(plan.W[3], 1), "predicted skew after the union");
    check(plan.pred == std::vector<double>({5,4,1,1,1,1,0,0}) &&
          plan.cooldown_until == std::vector<uint32_t>({0,3,0,3,0,0,0,0}), "pred and cooldown returned for installation");
    // A node moved earlier in the window is never re-moved by balance: from the post-refine state
    // (v1 already on owner 2, W {7,2,4,0}, lightest owner 3) v3 is the only node admissible toward
    // owner 3; marking it moved leaves balance with no admissible node.
    {
        std::vector<uint32_t> m = {0,2,0,0,1,1,2,3}; std::vector<double> W = {7, 2, 4, 0};
        std::vector<uint32_t> cd(8, 0); cd[1] = 3;
        std::vector<uint8_t> mn(8, 0); mn[1] = 1; mn[3] = 1; std::vector<WindowBurstAction> a;
        auto out = window_repair_balance(config, 4, hist, {5,4,1,1,1,1,0,0}, ones, rows, 20, 1, m, W, cd, mn, a);
        check(out.triggered && near(out.skew_before, 7 / 3.25) && out.moved == 0 && a.empty() && near(W[0], 7),
              "moved_now excludes the only admissible node from balance");
        // Without the mark, the same state moves v3 exactly as the union plan did.
        std::vector<uint8_t> only_refine(8, 0); only_refine[1] = 1; std::vector<WindowBurstAction> b;
        std::vector<uint32_t> mv2 = {0,2,0,0,1,1,2,3}; std::vector<double> W2 = {7, 2, 4, 0}; std::vector<uint32_t> cd2(8, 0); cd2[1] = 3;
        check(window_repair_balance(config, 4, hist, {5,4,1,1,1,1,0,0}, ones, rows, 20, 1, mv2, W2, cd2, only_refine, b).moved == 1 &&
              owners_of(b) == std::vector<uint32_t>({3}), "the unmarked post-refine state moves v3");
    }
    // CONVERGED above the trigger re-plans identically; ABORTED calibrates only.
    auto again = window_repair_plan(config, 4, WindowRepairPhase::CONVERGED, 1, 1, 80, membership, rows, work, nb, hist, ones, {}, {});
    check(again.valid && again.planned && again.actions.size() == 2, "converged phase above the trigger plans again");
    auto aborted = window_repair_plan(config, 4, WindowRepairPhase::ABORTED, 1, 1, 80, membership, rows, work, nb, hist, ones, {}, {});
    check(aborted.valid && !aborted.planned && aborted.actions.empty() && near(aborted.calibration.skew, 11 / 3.25) &&
          aborted.pred == plan.pred && near(aborted.W[0], 11), "aborted phase keeps calibrating without moves");
    // CONVERGED at or below the trigger: balanced fixture {0,0,1,1,2,2,3,3} with unit work -> skew 1.
    auto quiet = window_repair_plan(config, 4, WindowRepairPhase::CONVERGED, 1, 1, 80, {0,0,1,1,2,2,3,3}, rows,
                                    std::vector<uint64_t>(8, 1), nb, hist, ones, {}, {});
    check(quiet.valid && !quiet.planned && near(quiet.calibration.skew, 1.0), "converged phase below the trigger calibrates only");
    // Shape errors fail closed.
    check(!window_repair_plan(config, 4, WindowRepairPhase::IDLE, 1, 1, 80, membership, rows, work, nb, std::vector<uint32_t>(31, 0), ones, {}, {}).valid,
          "histogram shape mismatch is invalid");
    std::printf("test_plan_union_and_phases OK\n");
}

// The action history's rejection memory reaches the planner as a per-node target
// mask. An all-open mask plans exactly what no mask plans; a closed node or
// direction is never proposed by refine or balance. Same fixture as above.
void test_allowed_targets() {
    const std::vector<uint32_t> membership = {0,0,0,0,1,1,2,3};
    const std::vector<uint64_t> work = {5,4,1,1,1,1,0,0}, nb(8, 0), rows(8, 10);
    const auto hist = hist_of({{3,0,0,0}, {0,0,3,0}, {2,0,0,0}, {1,0,0,1}, {0,2,0,0}, {0,1,1,0}, {0,0,1,0}, {0,0,0,1}});
    const std::vector<double> ones = {1, 1, 1, 1};
    WindowRepairConfig config; config.enabled = true; config.refine_budget = 0.25; config.balance_budget = 0.25;
    config.eps = 1.0; config.tau = 0.2; config.cooldown_windows = 2; config.skew_trigger = 1.05;
    config.weight = WindowRepairWeight::WNB;  // nb is passed (all zero)
    const auto free_plan = window_repair_plan(config, 4, WindowRepairPhase::IDLE, 1, 1, 80, membership, rows, work, nb, hist,
                                              ones, {}, {});
    std::vector<uint64_t> open(8);
    for (size_t v = 0; v < 8; ++v) open[v] = uint64_t(15u ^ (1u << membership[v]));
    const auto open_plan = window_repair_plan(config, 4, WindowRepairPhase::IDLE, 1, 1, 80, membership, rows, work, nb, hist,
                                              ones, {}, {}, &open);
    check(open_plan.valid && owners_of(open_plan.actions) == owners_of(free_plan.actions) &&
          open_plan.actions.size() == 2 && open_plan.actions[0].target_owner == free_plan.actions[0].target_owner &&
          open_plan.actions[1].target_owner == free_plan.actions[1].target_owner &&
          open_plan.cooldown_until == free_plan.cooldown_until, "an all-open mask plans exactly the unmasked burst");
    // Node 1 cooled down: refine loses its only candidate; no other node is admissible toward owner 2.
    auto cooled = open; cooled[1] = 0;
    const auto without_v1 = window_repair_plan(config, 4, WindowRepairPhase::IDLE, 1, 1, 80, membership, rows, work, nb, hist,
                                               ones, {}, {}, &cooled);
    check(without_v1.valid && without_v1.stats.refine_candidates == 0 && without_v1.actions.empty(),
          "a cooled-down node is never proposed");
    // Direction v3: 0 -> 3 backed off: refine still moves v1, balance has no admissible node toward 3.
    auto backed_off = open; backed_off[3] &= ~(uint64_t{1} << 3);
    const auto without_v3 = window_repair_plan(config, 4, WindowRepairPhase::IDLE, 1, 1, 80, membership, rows, work, nb, hist,
                                               ones, {}, {}, &backed_off);
    check(without_v3.valid && without_v3.actions.size() == 1 && without_v3.actions[0].node == 1 &&
          without_v3.stats.balance_triggered && without_v3.stats.balance_moved == 0,
          "a backed-off direction is never proposed");
    std::vector<uint64_t> short_mask(7, 15);
    check(!window_repair_plan(config, 4, WindowRepairPhase::IDLE, 1, 1, 80, membership, rows, work, nb, hist, ones, {}, {},
                              &short_mask).valid, "a mask of the wrong size fails closed");
    std::printf("test_allowed_targets OK\n");
}

void test_verdict() {
    WindowRepairConfig config; config.enabled = true;  // slack .03, hop_slack .02
    config.recall_guard = true; config.recall_guard_pp = 0.10;
    const std::vector<double> base = {100, 80, 90, 70};   // peak 100, total 340
    const std::vector<double> ok = {101, 80, 90, 70};     // peak 1.01, total 341/340
    auto verdict = window_repair_verdict(config, 4, base, ok, 3.0, 3.03, 90.0, 90.0);
    check(verdict.accepted && near(verdict.peak_ratio, 1.01) && near(verdict.total_ratio, 341.0 / 340) && near(verdict.hop_ratio, 1.01),
          "all four conjuncts hold");
    check(window_repair_verdict(config, 4, base, ok, 3.0, 3.03, 89.9, 90.0).accepted, "recall at reference - guard passes");
    check(!window_repair_verdict(config, 4, base, ok, 3.0, 3.03, 89.89, 90.0).accepted, "recall below the guard fails");
    check(!window_repair_verdict(config, 4, base, ok, 3.0, 3.03, std::nan(""), 90.0).accepted &&
          !window_repair_verdict(config, 4, base, ok, 3.0, 3.03, 90.0, std::nan("")).accepted, "missing recall or reference fails");
    WindowRepairConfig unguarded = config; unguarded.recall_guard = false;
    check(window_repair_verdict(unguarded, 4, base, ok, 3.0, 3.03, std::nan(""), std::nan("")).accepted &&
          window_repair_verdict(unguarded, 4, base, ok, 3.0, 3.03, 10.0, 90.0).accepted,
          "without the guard recall never enters the verdict");
    check(!window_repair_verdict(unguarded, 4, base, ok, 3.0, 3.07, std::nan(""), std::nan("")).accepted,
          "without the guard the hop clause still applies");
    check(!window_repair_verdict(config, 4, base, ok, 3.0, 3.07, 90.0, 90.0).accepted, "hop ratio 1.0233 > 1.02 fails");
    check(!window_repair_verdict(config, 4, base, {104, 80, 90, 70}, 3.0, 3.0, 90.0, 90.0).accepted, "peak ratio 1.04 > 1.03 fails");
    const std::vector<double> total_up = {100, 100, 100, 55};  // peak ratio 1.0, total 355/340 = 1.044
    const auto total = window_repair_verdict(config, 4, base, total_up, 3.0, 3.0, 90.0, 90.0);
    check(!total.accepted && near(total.peak_ratio, 1.0) && near(total.total_ratio, 355.0 / 340), "total ratio alone fails");
    // Boundary with exactly representable slacks: peak 150/100 and hop 4.5/3 equal 1 + 0.5.
    WindowRepairConfig wide = config; wide.slack = 0.5; wide.hop_slack = 0.5;
    check(window_repair_verdict(wide, 4, base, {150, 80, 90, 70}, 3.0, 4.5, 90.0, 90.0).accepted, "ratios exactly at 1 + slack pass");
    check(!window_repair_verdict(wide, 4, base, {150, 80, 90, 70}, 3.0, 4.6, 90.0, 90.0).accepted, "hop ratio just above 1 + hop_slack fails");
    check(!window_repair_verdict(config, 4, {0, 0, 0, 0}, ok, 3.0, 3.0, 90.0, 90.0).accepted &&
          !window_repair_verdict(config, 4, base, ok, 0.0, 3.0, 90.0, 90.0).accepted, "zero baseline fails closed");
    std::printf("test_verdict OK\n");
}

void test_caps_identity_and_token() {
    WindowRepairConfig config; config.enabled = true;  // refine .01, balance .004, cum .45
    check(window_repair_total_cap(config, 1000000) == 900000, "total cap = 2 * .45 * total");
    check(window_repair_move_rate_cap(config, 1000000) == 112000, "rate cap = 8 * (.01 + .004) * total");
    config.cumulative_cap = 0.75;
    check(window_repair_total_cap(config, 1000000) == 1500000, "total cap = 2 * .75 * total (gross, above the table)");
    config.cumulative_cap = 1.0;
    check(window_repair_total_cap(config, 1000000) == 2000000, "cum 1 admits twice the table in gross bytes");
    WindowRepairConfig exact; exact.refine_budget = 0.5; exact.balance_budget = 0.25; exact.cumulative_cap = 0.125;
    check(window_repair_identity_tag(exact) == ":repair:local:refine=0.5:balance=0.25:cum=0.125:recall_guard=off:weight=w",
          "identity tag");
    exact.recall_guard = true; exact.recall_guard_pp = 0.5;
    check(window_repair_identity_tag(exact) == ":repair:local:refine=0.5:balance=0.25:cum=0.125:recall_guard=0.5:weight=w",
          "identity tag with the guard");
    exact.recall_guard = false; exact.recall_guard_pp = 0.0;
    exact.weight = WindowRepairWeight::WNB;
    check(window_repair_identity_tag(exact) == ":repair:local:refine=0.5:balance=0.25:cum=0.125:recall_guard=off:weight=wnb",
          "identity tag with weight wnb");
    exact.weight = WindowRepairWeight::W;
    WindowRepairConfig parsed; std::string error;
    check(window_repair_parse_token("algo:local,refine:0.02,balance:0.005,tau:0.3,eps:0.05,skew:1.1,cooldown:3,cum:0.4,"
                                    "slack:0.04,hop_slack:0.03,recall_guard:0.2,ema:0.6,converge:4,guard:0.15", &parsed, &error) &&
          parsed.enabled && near(parsed.refine_budget, 0.02) && near(parsed.balance_budget, 0.005) && near(parsed.tau, 0.3) &&
          near(parsed.eps, 0.05) && near(parsed.skew_trigger, 1.1) && parsed.cooldown_windows == 3 && near(parsed.cumulative_cap, 0.4) &&
          near(parsed.slack, 0.04) && near(parsed.hop_slack, 0.03) && parsed.recall_guard && near(parsed.recall_guard_pp, 0.2) &&
          near(parsed.ema, 0.6) &&
          parsed.converge_windows == 4 && near(parsed.guard, 0.15), "every key parses into the config");
    WindowRepairConfig defaults;
    check(window_repair_parse_token("algo:local", &defaults, &error) && defaults.enabled && near(defaults.refine_budget, 0.01) &&
          near(defaults.balance_budget, 0.004) && defaults.cooldown_windows == 2 && defaults.converge_windows == 3 &&
          !defaults.recall_guard && defaults.weight == WindowRepairWeight::W,
          "algo alone keeps the defaults; the recall guard is off; the weight is W");
    check(window_repair_config_json(defaults).find(",\"weight\":\"w\"}") != std::string::npos, "config JSON says weight w");
    // weight: parsed before the numeric values, w or wnb only, once.
    WindowRepairConfig weighted;
    check(window_repair_parse_token("algo:local,weight:wnb,refine:0.03", &weighted, &error) &&
          weighted.weight == WindowRepairWeight::WNB && near(weighted.refine_budget, 0.03) &&
          window_repair_config_json(weighted).find(",\"weight\":\"wnb\"}") != std::string::npos,
          "weight:wnb parses (and the JSON says so)");
    check(window_repair_parse_token("algo:local,weight:w", &weighted, &error) && weighted.weight == WindowRepairWeight::W,
          "weight:w parses");
    for (const char* token : {"algo:local,weight:x", "algo:local,weight:1", "algo:local,weight:WNB", "algo:local,weight:nb",
                              "algo:local,weight:wnb,weight:w", "algo:local,weight:w,weight:w", "algo:local,weight:"}) {
        WindowRepairConfig refused;
        check(!window_repair_parse_token(token, &refused, &error) && !refused.enabled, "a bad or repeated weight is refused");
    }
    WindowRepairConfig refused;
    check(!window_repair_parse_token("algo:local,weight:x", &refused, &error) &&
          error.find("weight must be w (the default) or wnb") != std::string::npos, "weight:x names the domain");
    check(!window_repair_parse_token("algo:local,weight:wnb,weight:wnb", &refused, &error) &&
          error.find("duplicate key weight") != std::string::npos, "a duplicate weight is a duplicate key");
    WindowRepairConfig off;
    check(window_repair_parse_token("algo:local,recall_guard:off", &off, &error) && !off.recall_guard,
          "recall_guard:off is explicit");
    check(window_repair_config_json(off).find("\"recall_guard\":\"off\"") != std::string::npos, "config JSON says off");
    WindowRepairConfig zero;
    check(window_repair_parse_token("algo:local,recall_guard:0", &zero, &error) && zero.recall_guard &&
          zero.recall_guard_pp == 0.0, "recall_guard:0 is a guard with no tolerance");
    WindowRepairConfig bad_guard;
    check(!window_repair_parse_token("algo:local,recall_guard:-1", &bad_guard, &error) &&
          !window_repair_parse_token("algo:local,recall_guard:on", &bad_guard, &error) &&
          !window_repair_parse_token("algo:local,recall_guard:off,recall_guard:0.1", &bad_guard, &error),
          "negative, non-numeric and duplicate guards are refused");
    WindowRepairConfig bad;
    check(!window_repair_parse_token("refine:0.01", &bad, &error) && error.find("algo") != std::string::npos, "algo is required");
    check(!window_repair_parse_token("algo:global", &bad, &error), "only the local algorithm is known");
    check(!window_repair_parse_token("algo:local,refine:0.01,refine:0.02", &bad, &error) && error.find("duplicate") != std::string::npos,
          "duplicate key rejected");
    check(!window_repair_parse_token("algo:local,budget:0.01", &bad, &error) && error.find("unknown") != std::string::npos, "unknown key rejected");
    check(!window_repair_parse_token("algo:local,skew:0.9", &bad, &error) && !window_repair_parse_token("algo:local,ema:0", &bad, &error) &&
          !window_repair_parse_token("algo:local,cum:1.5", &bad, &error) && !window_repair_parse_token("algo:local,converge:0", &bad, &error) &&
          !window_repair_parse_token("algo:local,cooldown:1.5", &bad, &error) && !window_repair_parse_token("algo:local,eps:x", &bad, &error) &&
          !window_repair_parse_token("algo:local,", &bad, &error) && !window_repair_parse_token("", &bad, &error) && !bad.enabled,
          "domain and syntax failures leave the output disabled");
    check(window_repair_config_json(exact).rfind(",\"repair_config\":{\"algo\":\"local\",\"refine\":0.5,\"balance\":0.25", 0) == 0,
          "config JSON echoes the parsed values");
    // balance_target and verify_invalid_limit.
    WindowRepairConfig target;
    check(window_repair_parse_token("algo:local", &target, &error) &&
          target.balance_target == WindowRepairBalanceTarget::LIGHTEST && target.verify_invalid_limit == 3,
          "defaults: balance_target lightest (at every K), verify_invalid_limit 3");
    check(window_repair_parse_token("algo:local,balance_target:underloaded,verify_invalid_limit:5", &target, &error) &&
          target.balance_target == WindowRepairBalanceTarget::UNDERLOADED && target.verify_invalid_limit == 5,
          "balance_target and verify_invalid_limit parse");
    check(window_repair_parse_token("algo:local,balance_target:lightest", &target, &error) &&
          target.balance_target == WindowRepairBalanceTarget::LIGHTEST, "balance_target:lightest");
    WindowRepairConfig bad_target;
    check(!window_repair_parse_token("algo:local,balance_target:auto", &bad_target, &error) &&
          error.find("lightest (the default) or underloaded") != std::string::npos && !bad_target.enabled,
          "balance_target:auto is refused");
    check(!window_repair_parse_token("algo:local,balance_target:heaviest", &bad_target, &error) &&
          error.find("balance_target") != std::string::npos &&
          !window_repair_parse_token("algo:local,balance_target:lightest,balance_target:underloaded", &bad_target,
                                     &error) &&
          !window_repair_parse_token("algo:local,verify_invalid_limit:0", &bad_target, &error) &&
          !window_repair_parse_token("algo:local,verify_invalid_limit:2.5", &bad_target, &error) && !bad_target.enabled,
          "an unknown or duplicate balance target and a limit below 1 or not whole are refused");
    check(window_repair_parse_token("algo:local,balance_target:lightest", &target, &error) &&
          window_repair_config_json(target).find(
              "\"balance_target\":\"lightest\",\"verify_invalid_limit\":3,\"weight\":\"w\"}") != std::string::npos,
          "config JSON carries the balance target, the invalid-window limit and (last) the weight");
    std::printf("test_caps_identity_and_token OK\n");
}

void test_convergence_and_guard() {
    WindowRepairConfig config; config.enabled = true;  // trigger 1.05, converge 3, guard .10
    WindowRepairState state;
    auto t = window_repair_account_window(config, state, false, 1.02);
    check(state.phase == WindowRepairPhase::ACTIVE && state.windows == 1 && state.quiet_windows == 1 && !t.converged && !t.reactivated,
          "first window activates and counts a quiet window");
    window_repair_account_window(config, state, false, 1.01);
    check(state.quiet_windows == 2, "second quiet window");
    window_repair_account_window(config, state, true, 1.01);
    check(state.quiet_windows == 0 && state.phase == WindowRepairPhase::ACTIVE, "an issued burst resets the quiet count");
    window_repair_account_window(config, state, false, 1.2);
    check(state.quiet_windows == 0, "no burst but skew above the trigger is not quiet");
    window_repair_account_window(config, state, false, 1.0);
    window_repair_account_window(config, state, false, 1.0);
    t = window_repair_account_window(config, state, false, 1.05);
    check(t.converged && state.phase == WindowRepairPhase::CONVERGED && state.windows == 7 && state.quiet_windows == 3,
          "three quiet windows converge (skew equal to the trigger is quiet)");
    t = window_repair_account_window(config, state, false, 1.03);
    check(!t.converged && !t.reactivated && state.phase == WindowRepairPhase::CONVERGED && state.windows == 8, "converged windows keep calibrating");
    t = window_repair_account_window(config, state, true, 1.2);
    check(t.reactivated && state.phase == WindowRepairPhase::ACTIVE && state.quiet_windows == 0 && near(state.skew, 1.2),
          "skew above the trigger re-activates repair");
    state.phase = WindowRepairPhase::ABORTED;
    t = window_repair_account_window(config, state, false, 1.0);
    check(!t.converged && !t.reactivated && state.phase == WindowRepairPhase::ABORTED, "aborted repair never changes phase");
    // Cumulative guard on the last converge_windows verified peaks against (1 + guard) * first_peak.
    WindowRepairState guard; guard.first_peak_us = 100.0;
    check(!window_repair_guard_tripped(config, guard, 111.0) && !window_repair_guard_tripped(config, guard, 112.0),
          "fewer than converge_windows verified windows never trip");
    check(!window_repair_guard_tripped(config, guard, 105.0), "one verified window within the guard keeps repair alive");
    check(!window_repair_guard_tripped(config, guard, 111.0) && !window_repair_guard_tripped(config, guard, 112.0) &&
          window_repair_guard_tripped(config, guard, 113.0) && guard.verified_peaks.size() == 3,
          "three consecutive verified windows above 110 trip the guard");
    WindowRepairState boundary; boundary.first_peak_us = 100.0;
    check(!window_repair_guard_tripped(config, boundary, 110.0) && !window_repair_guard_tripped(config, boundary, 110.0) &&
          !window_repair_guard_tripped(config, boundary, 110.0), "exactly (1 + guard) * first is not a regression");
    WindowRepairState unknown;
    check(!window_repair_guard_tripped(config, unknown, 200.0) && !window_repair_guard_tripped(config, unknown, 200.0) &&
          !window_repair_guard_tripped(config, unknown, 200.0), "without a first-window peak the guard cannot fire");
    std::printf("test_convergence_and_guard OK\n");
}
// ---------------------------------------------------------------------------
// Randomized equality with the reference (test_window_repair_reference.h). The
// planner in this test is compiled with the production flags, like
// tide_coordinator's (floating-point contraction on: GCC fuses a multiply that
// feeds an add within one basic block); only the reference's own source file
// carries -ffp-contract=off (CMakeLists.txt). Every decision (actions, counts,
// cooldowns) must agree exactly and every load within 1e-12 x the trial's load
// scale. A differing decision passes only as a ROUNDING BOUNDARY: the exact
// reference took that same decision, at the same action count, on a comparison
// whose two sides were within that tolerance (a fused multiply-add may tip it
// either way). Each boundary case is printed with its first difference and the
// near tie; they must stay rare (at most 1 % of the compared plans).
// ---------------------------------------------------------------------------
using repair_reference::Agreement;
using repair_reference::Comparison;

struct RandomizedCounts {
    size_t compared = 0, exact = 0, bit_identical = 0, boundary = 0;
    double max_load_diff = 0.0;
};
void account(RandomizedCounts& counts, const Comparison& c, const std::string& where) {
    ++counts.compared;
    if (c.agreement == Agreement::EXACT) {
        ++counts.exact;
        counts.bit_identical += c.bit_identical ? 1 : 0;
        counts.max_load_diff = std::max(counts.max_load_diff, c.max_load_diff);
        return;
    }
    if (c.agreement == Agreement::ROUNDING_BOUNDARY) {
        ++counts.boundary;
        std::printf("  rounding boundary at %s: %s; near tie: %s\n", where.c_str(), c.difference.c_str(),
                    c.boundary.c_str());
        return;
    }
    std::fprintf(stderr, "planner/reference mismatch at %s: %s\n", where.c_str(), c.difference.c_str());
    std::abort();
}
void check_rare(const RandomizedCounts& counts, const char* what) {
    check(counts.boundary * 100 <= counts.compared, what);
}

const WindowRepairBalanceTarget kTargets[] = {WindowRepairBalanceTarget::LIGHTEST,
                                             WindowRepairBalanceTarget::UNDERLOADED};

// Randomised equality of window_repair_plan (calibrate, refine, the
// admissible-list LIGHTEST walk or the UNDERLOADED walk) with the reference for
// K in {2, 3, 4, 8, 16}: skewed memberships, tied integer weights, equal and
// mixed prices, cooldown rows, a smoothed second window, optional target masks,
// both balance targets, tau 0..0.5, eps 0..1 and budgets from nothing to
// everything. K = 1 is refused everywhere (test_owner_limits covers K = 65).
void test_plan_matches_reference_for_k() {
    std::mt19937_64 rng(12);
    auto uni = [&](uint64_t n) { return uint64_t(rng() % n); };
    const double taus[] = {0.0, 0.1, 0.2, 0.3, 0.5}, epss[] = {0.0, 0.03, 0.1, 1.0};
    const uint32_t ks[] = {2, 3, 4, 8, 16};
    size_t refine_moves = 0, balance_moves = 0, underloaded_moves = 0, weight_w_moves = 0, weight_w_plans = 0;
    RandomizedCounts counts;
    for (const uint32_t K : ks) {
        size_t k_moves = 0;
        for (int trial = 0; trial < 1500; ++trial) {
            const size_t n = 1 + uni(trial % 4 == 0 ? 500 : 80);
            WindowRepairConfig config; config.enabled = true; config.tau = taus[uni(5)]; config.eps = epss[uni(4)];
            config.skew_trigger = 1.0 + 0.05 * double(uni(3)); config.cooldown_windows = uint32_t(uni(3));
            config.refine_budget = uni(4) == 0 ? 0.0 : 0.02 * double(1 + uni(25));
            config.balance_budget = uni(4) == 0 ? 0.0 : 0.02 * double(1 + uni(25));
            config.balance_target = kTargets[uni(2)];
            config.weight = WindowRepairWeight::WNB;  // nb is drawn; the W plan below reuses the inputs without it
            std::vector<uint32_t> membership(n), cooldown(n, 0), hist(n * K);
            std::vector<uint64_t> work(n), nb(n), rows(n);
            const uint32_t heavy = uint32_t(uni(K));
            for (size_t v = 0; v < n; ++v) {
                membership[v] = uni(3) == 0 ? heavy : uint32_t(uni(K));
                work[v] = uni(4) == 0 ? 0 : uni(12);
                nb[v] = uni(3) == 0 ? 0 : uni(20);
                rows[v] = 4 + 4 * uni(60);
                for (uint32_t p = 0; p < K; ++p) hist[v * K + p] = uni(3) == 0 ? 0u : uint32_t(uni(6));
                if (uni(10) == 0) cooldown[v] = 3 + uint32_t(uni(3));
            }
            std::vector<double> price(K);
            for (auto& p : price) p = uni(2) ? 1.0 : 0.75 + 0.05 * double(uni(12));
            uint64_t total = 0;
            for (const auto r : rows) total += r;
            std::vector<double> previous;
            if (uni(3) == 0) { previous.resize(n); for (auto& x : previous) x = double(uni(9)) * 0.5; }
            std::vector<uint64_t> mask;
            if (uni(3) == 0) {
                mask.resize(n);
                for (auto& m : mask) m = uni(2) ? ~uint64_t{0} : rng();
            }
            const auto* allowed = mask.empty() ? nullptr : &mask;
            const uint64_t queries = 1 + uni(64);
            const auto plan = window_repair_plan(config, K, WindowRepairPhase::IDLE, 3, queries, total, membership, rows,
                                                 work, nb, hist, price, previous, cooldown, allowed);
            repair_reference::ReferenceTrace trace;
            const auto ref = repair_reference::exact_plan(config, K, 3, queries, total, membership, rows, work, nb,
                                                          hist, price, previous, cooldown, allowed, &trace);
            for (const auto& action : plan.actions) check(action.target_owner < K && action.source_owner < K, "owners < K");
            char where[160];
            std::snprintf(where, sizeof where, "K %u trial %d (n %zu, tau %g, eps %g, target %s)", K, trial, n,
                          config.tau, config.eps, window_repair_balance_target_name(config.balance_target));
            account(counts, repair_reference::compare_plan(plan, ref, trace, config.balance_target), where);
            refine_moves += plan.stats.refine_moved; balance_moves += plan.stats.balance_moved;
            if (plan.stats.balance_target == WindowRepairBalanceTarget::UNDERLOADED) underloaded_moves += plan.stats.balance_moved;
            k_moves += plan.actions.size();
            // The same inputs under weight W (the default): no nbwork, pred = work / Q. No extra
            // random draw, so the WNB sequence above is unchanged.
            WindowRepairConfig w_config = config; w_config.weight = WindowRepairWeight::W;
            const std::vector<uint64_t> no_nb;
            const auto plan_w = window_repair_plan(w_config, K, WindowRepairPhase::IDLE, 3, queries, total, membership,
                                                   rows, work, no_nb, hist, price, previous, cooldown, allowed);
            check(plan_w.valid, "a weight-W plan without nbwork is valid");
            for (size_t v = 0; v < n && previous.size() != n; ++v)
                check(plan_w.pred[v] == double(work[v]) / double(queries), "weight W: first-window pred = w / Q");
            repair_reference::ReferenceTrace trace_w;
            const auto ref_w = repair_reference::exact_plan(w_config, K, 3, queries, total, membership, rows, work,
                                                            no_nb, hist, price, previous, cooldown, allowed, &trace_w);
            account(counts, repair_reference::compare_plan(plan_w, ref_w, trace_w, config.balance_target),
                    std::string(where) + " weight w");
            weight_w_moves += plan_w.actions.size(); ++weight_w_plans;
        }
        check(k_moves > 500, "every K exercises many moves");
    }
    check(refine_moves > 1000 && balance_moves > 1000 && underloaded_moves > 500,
          "both rules and both balance targets move many nodes across the K values");
    check(weight_w_plans == 7500 && weight_w_moves > 1000, "the weight-W plans move many nodes too");
    check_rare(counts, "rounding-boundary cases stay rare (<= 1 % of the plans)");
    // K = 1 is refused: refine needs a best OTHER owner.
    WindowRepairConfig config; config.enabled = true; config.weight = WindowRepairWeight::WNB;
    const std::vector<uint32_t> one_owner = {0, 0};
    std::vector<double> pred, W;
    WindowRepairCalibration calibration; std::string error;
    check(!window_repair_calibrate(config, 1, {1, 2}, {0, 0}, 1, one_owner, {1.0}, pred, W, &calibration, &error) &&
          error.find("owner count") != std::string::npos, "K = 1 calibration refused");
    const auto one = window_repair_plan(config, 1, WindowRepairPhase::IDLE, 1, 1, 8, one_owner, {4, 4}, {1, 2}, {0, 0},
                                        {1, 1}, {1.0}, {}, {});
    check(!one.valid && one.error.find("owner count") != std::string::npos && one.actions.empty(), "K = 1 plan refused");
    std::vector<uint32_t> m = one_owner, cd(2, 0); std::vector<uint8_t> mn(2, 0);
    std::vector<double> w1 = {3.0}; std::vector<WindowBurstAction> a;
    check(window_repair_refine(config, 1, {1, 1}, {1, 2}, {1.0}, {4, 4}, 100, 1, m, w1, cd, mn, a).candidates == 0 &&
          !window_repair_balance(config, 1, {1, 1}, {1, 2}, {1.0}, {4, 4}, 100, 1, m, w1, cd, mn, a).triggered &&
          a.empty(), "K = 1 refine and balance do nothing");
    check(!window_repair_verdict(config, 1, {1.0}, {1.0}, 1.0, 1.0, 90.0, 90.0).accepted, "K = 1 verdict fails closed");
    check(!window_repair_verdict(config, 8, {1, 1, 1, 1}, {1, 1, 1, 1}, 1.0, 1.0, 90.0, 90.0).accepted,
          "cost vectors of another size than K fail closed");
    check(window_repair_verdict(config, 8, std::vector<double>(8, 1.0), std::vector<double>(8, 1.0), 1.0, 1.0,
                                90.0, 90.0).accepted, "a K = 8 verdict compares eight ranks");
    std::printf("test_plan_matches_reference_for_k OK (%zu plans, half of them weight W: %zu with identical decisions, "
                "%zu of them bit for bit, max load difference %.3g x scale; %zu rounding boundaries; WNB: %zu refine, "
                "%zu balance and %zu underloaded moves; W: %zu moves)\n", counts.compared, counts.exact,
                counts.bit_identical, counts.max_load_diff, counts.boundary, refine_moves, balance_moves,
                underloaded_moves, weight_w_moves);
}

// The balance step alone against the reference, with rows pre-marked as moved
// earlier in the window and byte budgets anywhere from one byte to the total.
// Part 1: 3,000 K = 4 cases of the LIGHTEST walk against the reference
// (seed 13). Part 2: 3,000 cases of the UNDERLOADED walk at K in
// {3, 4, 8, 16}, with target masks. Same agreement rule as above.
void test_balance_matches_reference() {
    std::mt19937_64 rng(13);
    auto uni = [&](uint64_t n) { return uint64_t(rng() % n); };
    const double taus[] = {0.0, 0.1, 0.2, 0.3, 0.5};
    size_t compared_moves = 0;
    RandomizedCounts lightest, underloaded;
    auto compare = [&](RandomizedCounts& counts, const WindowRepairConfig& config, uint32_t K,
                       const std::vector<uint32_t>& hist, const std::vector<double>& pred, const std::vector<double>& price,
                       const std::vector<uint64_t>& rows, uint64_t budget, const std::vector<uint32_t>& membership,
                       const std::vector<uint32_t>& cooldown, const std::vector<uint8_t>& moved,
                       const std::vector<double>& W, const std::vector<uint64_t>* allowed, const std::string& where) {
        auto mv1 = membership, mv2 = membership;
        auto c1 = cooldown, c2 = cooldown;
        auto d1 = moved, d2 = moved;
        auto W1 = W, W2 = W;
        std::vector<WindowBurstAction> a1, a2;
        const auto r1 = window_repair_balance(config, K, hist, pred, price, rows, budget, 3, mv1, W1, c1, d1, a1, allowed);
        repair_reference::ReferenceTrace trace;
        const auto r2 = repair_reference::exact_balance(config, K, hist, pred, price, rows, budget, 3, mv2, W2, c2, d2,
                                                        a2, allowed, &trace);
        check(r1.target == config.balance_target, "the balance step applies the configured target");
        account(counts, repair_reference::compare_balance({&r1, &mv1, &c1, &d1, &W1, &a1}, {&r2, &mv2, &c2, &d2, &W2, &a2},
                                                          trace), where);
        return a1.size();
    };
    for (int trial = 0; trial < 3000; ++trial) {
        const size_t n = 1 + uni(trial % 3 == 0 ? 400 : 60);
        WindowRepairConfig config; config.enabled = true; config.tau = taus[uni(5)];
        config.skew_trigger = 1.0 + 0.05 * double(uni(3)); config.cooldown_windows = uint32_t(uni(3));
        config.balance_target = WindowRepairBalanceTarget::LIGHTEST;
        std::vector<uint32_t> membership(n), cooldown(n, 0), hist(n * 4);
        std::vector<uint8_t> moved(n, 0);
        std::vector<double> pred(n);
        std::vector<uint64_t> rows(n);
        const uint32_t heavy = uint32_t(uni(4));
        for (size_t v = 0; v < n; ++v) {
            membership[v] = uni(3) == 0 ? heavy : uint32_t(uni(4));
            pred[v] = uni(4) == 0 ? 0.0 : double(uni(12)) * (uni(2) ? 1.0 : 0.25);
            rows[v] = 1 + uni(200);
            for (uint32_t p = 0; p < 4; ++p) hist[v * 4 + p] = uni(3) == 0 ? 0u : uint32_t(uni(6));
            if (uni(10) == 0) cooldown[v] = 3 + uint32_t(uni(3));
            if (uni(15) == 0) moved[v] = 1;
        }
        std::vector<double> price(4);
        for (auto& p : price) p = uni(2) ? 1.0 : 0.75 + 0.05 * double(uni(12));
        std::vector<double> W(4, 0.0);
        for (size_t v = 0; v < n; ++v) W[membership[v]] += price[membership[v]] * pred[v];
        uint64_t total = 0;
        for (const auto r : rows) total += r;
        const uint64_t budget = uni(4) == 0 ? total : 1 + uni(total);
        compared_moves += compare(lightest, config, 4, hist, pred, price, rows, budget, membership, cooldown, moved, W,
                                  nullptr, "lightest balance trial " + std::to_string(trial));
    }
    check(compared_moves > 1000, "the randomised K = 4 cases exercise many LIGHTEST moves");
    check_rare(lightest, "LIGHTEST rounding-boundary cases stay rare");
    const double epss[] = {0.0, 0.03, 0.1, 0.5, 1.0};
    const uint32_t ks[] = {3, 4, 8, 16};
    size_t underloaded_moves = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const uint32_t K = ks[trial % 4];
        const size_t n = 1 + uni(trial % 3 == 0 ? 400 : 60);
        WindowRepairConfig config; config.enabled = true; config.tau = taus[uni(5)]; config.eps = epss[uni(5)];
        config.skew_trigger = 1.0 + 0.05 * double(uni(3)); config.cooldown_windows = uint32_t(uni(3));
        config.balance_target = WindowRepairBalanceTarget::UNDERLOADED;
        std::vector<uint32_t> membership(n), cooldown(n, 0), hist(n * K);
        std::vector<uint8_t> moved(n, 0);
        std::vector<double> pred(n);
        std::vector<uint64_t> rows(n);
        const uint32_t heavy = uint32_t(uni(K)), second = uint32_t(uni(K));
        for (size_t v = 0; v < n; ++v) {
            const uint64_t pick = uni(4);
            membership[v] = pick == 0 ? heavy : pick == 1 ? second : uint32_t(uni(K));
            pred[v] = uni(4) == 0 ? 0.0 : double(uni(12)) * (uni(2) ? 1.0 : 0.25);
            rows[v] = 1 + uni(200);
            for (uint32_t p = 0; p < K; ++p) hist[v * K + p] = uni(3) == 0 ? 0u : uint32_t(uni(6));
            if (uni(10) == 0) cooldown[v] = 3 + uint32_t(uni(3));
            if (uni(15) == 0) moved[v] = 1;
        }
        std::vector<double> price(K);
        for (auto& p : price) p = uni(2) ? 1.0 : 0.75 + 0.05 * double(uni(12));
        std::vector<double> W(K, 0.0);
        for (size_t v = 0; v < n; ++v) W[membership[v]] += price[membership[v]] * pred[v];
        uint64_t total = 0;
        for (const auto r : rows) total += r;
        const uint64_t budget = uni(4) == 0 ? total : 1 + uni(total);
        std::vector<uint64_t> mask;
        if (uni(3) == 0) {
            mask.resize(n);
            for (auto& m : mask) m = uni(2) ? ~uint64_t{0} : rng();
        }
        const auto* allowed = mask.empty() ? nullptr : &mask;
        underloaded_moves += compare(underloaded, config, K, hist, pred, price, rows, budget, membership, cooldown,
                                     moved, W, allowed,
                                     "underloaded balance trial " + std::to_string(trial) + " K " + std::to_string(K));
    }
    check(underloaded_moves > 500, "the randomised cases exercise many UNDERLOADED moves");
    check_rare(underloaded, "UNDERLOADED rounding-boundary cases stay rare");
    std::printf("test_balance_matches_reference OK (%zu LIGHTEST and %zu UNDERLOADED moves compared; boundaries %zu and "
                "%zu; identical decisions %zu/%zu and %zu/%zu, bit for bit %zu and %zu)\n", compared_moves,
                underloaded_moves, lightest.boundary, underloaded.boundary, lightest.exact, lightest.compared,
                underloaded.exact, underloaded.compared, lightest.bit_identical, underloaded.bit_identical);
}

// K = 8 by hand. Fixture D: unit prices, 10-byte rows, tau .3 (allow[v][p]:
// hist >= .3 x deg), skew trigger 1.05.
//   owner 0: v0 pred 3 {1,2,0..}  (admissible: 0, 1)     v1 pred 3 {1,0,2,0..} (0, 2)
//            v2 pred 2 {0,1,0,2,0..} (1, 3)              v3 pred 2 {3,0..}     (0 only)
//   owners 1..6: v4..v9, pred 1 each, every neighbour on its own owner; owner 7 is empty.
//   W = {10,1,1,1,1,1,1,0}: mean 2, skew 5.
struct FixtureD {
    std::vector<uint32_t> membership = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6};
    std::vector<double> pred = {3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    std::vector<double> price = std::vector<double>(8, 1.0);
    std::vector<uint64_t> rows = std::vector<uint64_t>(10, 10);
    std::vector<uint32_t> hist;
    FixtureD() {
        const uint32_t rows8[10][8] = {
            {1, 2, 0, 0, 0, 0, 0, 0}, {1, 0, 2, 0, 0, 0, 0, 0}, {0, 1, 0, 2, 0, 0, 0, 0}, {3, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 0}};
        for (const auto& row : rows8) hist.insert(hist.end(), row, row + 8);
    }
    WindowRepairConfig config(WindowRepairBalanceTarget target, double eps) const {
        WindowRepairConfig c; c.enabled = true; c.tau = 0.3; c.skew_trigger = 1.05; c.cooldown_windows = 1;
        c.eps = eps; c.balance_target = target; c.weight = WindowRepairWeight::WNB; return c;
    }
    WindowRepairBalanceResult run(const WindowRepairConfig& c, uint64_t budget, std::vector<uint32_t>& m,
                                  std::vector<double>& W, std::vector<WindowBurstAction>& a,
                                  std::vector<uint8_t> moved = std::vector<uint8_t>(10, 0)) const {
        m = membership; W = {10, 1, 1, 1, 1, 1, 1, 0}; a.clear();
        std::vector<uint32_t> cd(10, 0);
        return window_repair_balance(c, 8, hist, pred, price, rows, budget, 1, m, W, cd, moved, a);
    }
};

void test_balance_target_k8() {
    const FixtureD f;
    std::vector<uint32_t> m; std::vector<double> W; std::vector<WindowBurstAction> a;
    // LIGHTEST: heaviest 0, lightest 7 (empty), gap 10. No node of owner 0 has a
    // neighbour on owner 7, so the walk stops at once: nothing moves.
    auto lightest = f.run(f.config(WindowRepairBalanceTarget::LIGHTEST, 1.0), 1000, m, W, a);
    check(lightest.triggered && lightest.moved == 0 && a.empty() && near(W[0], 10) &&
          lightest.target == WindowRepairBalanceTarget::LIGHTEST,
          "lightest: the empty owner 7 admits no node of owner 0, so balance moves nothing");
    // UNDERLOADED, eps 1 (limit 2 x mean = 4; the mean stays 2 with unit prices):
    //   step 1: peak 0 (10); v0 -> 1 (1 + 3 = 4 <= 4, < 10)             W {7,4,1,1,1,1,1,0}
    //   step 2: peak 0 (7);  v1 -> 2 (1 + 3 = 4)                         W {4,4,4,1,1,1,1,0}
    //   step 3: peak 0 (4, first max); v2: owner 1 holds 4 (not below the mean),
    //           owner 3 (1 + 2 = 3 < 4) qualifies -> v2 -> 3                W {2,4,4,3,1,1,1,0}
    //   step 4: peak 1 (4); its only candidate v4 has no admissible other owner: stop.
    auto under = f.run(f.config(WindowRepairBalanceTarget::UNDERLOADED, 1.0), 1000, m, W, a);
    check(under.triggered && under.moved == 3 && under.bytes == 30 && owners_of(a) == std::vector<uint32_t>({0, 1, 2}) &&
          a[0].target_owner == 1 && a[1].target_owner == 2 && a[2].target_owner == 3 &&
          a[0].source_owner == 0 && a[2].source_owner == 0 && under.target == WindowRepairBalanceTarget::UNDERLOADED,
          "underloaded: balance keeps moving to the underloaded owners that hold the nodes' neighbours");
    check(near(W[0], 2) && near(W[1], 4) && near(W[2], 4) && near(W[3], 3) && near(W[7], 0) &&
          m[0] == 1 && m[1] == 2 && m[2] == 3 && m[3] == 0, "loads and membership after the underloaded walk");
    // The default (no balance_target in the token) is LIGHTEST at K = 8 too.
    WindowRepairConfig parsed; std::string error;
    check(window_repair_parse_token("algo:local,tau:0.3,skew:1.05,cooldown:1,eps:1", &parsed, &error), "token parses");
    auto by_default = f.run(parsed, 1000, m, W, a);
    check(by_default.target == WindowRepairBalanceTarget::LIGHTEST && by_default.moved == 0 && a.empty(),
          "the default balance target is lightest at K = 8");
    // eps .03: limit 2.06; every destination would exceed it (4, 4, 3, 3): nothing moves.
    auto tight = f.run(f.config(WindowRepairBalanceTarget::UNDERLOADED, 0.03), 1000, m, W, a);
    check(tight.triggered && tight.moved == 0 && a.empty(), "a destination never exceeds (1 + eps) x the mean");
    // Byte budget 15: the second move reaches it (20 >= 15) and ends the walk.
    auto capped = f.run(f.config(WindowRepairBalanceTarget::UNDERLOADED, 1.0), 15, m, W, a);
    check(capped.moved == 2 && capped.bytes == 20 && owners_of(a) == std::vector<uint32_t>({0, 1}),
          "the byte budget ends the underloaded walk after the crossing move");
    // v0 moved earlier this window: v1 -> 2 (W {7,1,4,..}), then v2 -> 3 (owner 3 holds more
    // of its neighbours than owner 1), then peak 0 (5) has only v3 left (no other owner): stop.
    std::vector<uint8_t> premarked(10, 0); premarked[0] = 1;
    auto marked = f.run(f.config(WindowRepairBalanceTarget::UNDERLOADED, 1.0), 1000, m, W, a, premarked);
    check(marked.moved == 2 && owners_of(a) == std::vector<uint32_t>({1, 2}) && a[1].target_owner == 3 &&
          near(W[0], 5) && near(W[2], 4) && near(W[3], 3), "a node moved earlier this window is never re-moved");
    // Destination choice. Fixture E, K = 8, unit prices, eps 10, tau .3:
    //   v0 owner 0 pred 4, neighbours {1,0,h2,h3,0..}; v1 owner 2 pred 1; v2 owner 3 pred .5; v3 owner 0 pred 4 (own only)
    //   W = {8,0,1,.5,0,0,0,0}, mean 1.1875: owners 2 and 3 are below the mean, owner 1 holds no neighbour.
    auto choose = [&](uint32_t h2, uint32_t h3) {
        const std::vector<uint32_t> hist = {
            1, 0, h2, h3, 0, 0, 0, 0,   0, 0, 1, 0, 0, 0, 0, 0,   0, 0, 0, 1, 0, 0, 0, 0,   4, 0, 0, 0, 0, 0, 0, 0};
        std::vector<uint32_t> mm = {0, 2, 3, 0}, cd(4, 0); std::vector<uint8_t> mv(4, 0);
        std::vector<double> WW = {8, 0, 1, 0.5, 0, 0, 0, 0}; std::vector<WindowBurstAction> aa;
        WindowRepairConfig c = f.config(WindowRepairBalanceTarget::UNDERLOADED, 10.0);
        const auto out = window_repair_balance(c, 8, hist, {4, 1, 0.5, 4}, std::vector<double>(8, 1.0),
                                               {10, 10, 10, 10}, 1000, 1, mm, WW, cd, mv, aa);
        return out.moved == 1 && aa.size() == 1 && aa[0].node == 0 ? int(aa[0].target_owner) : -1;
    };
    check(choose(2, 2) == 3, "equal neighbour counts: the lighter underloaded owner (3: .5 < 1) is the destination");
    check(choose(3, 2) == 2, "more neighbours win over a lighter load (owner 2 holds 3 of v0's neighbours)");
    // A plan at K = 8 through window_repair_plan reports the rule it applied.
    WindowRepairConfig c = f.config(WindowRepairBalanceTarget::UNDERLOADED, 1.0);
    c.refine_budget = 0.0; c.balance_budget = 1.0;
    const std::vector<uint64_t> work = {3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    const auto plan = window_repair_plan(c, 8, WindowRepairPhase::IDLE, 1, 1, 100, f.membership, f.rows, work,
                                         std::vector<uint64_t>(10, 0), f.hist, f.price, {}, {});
    check(plan.valid && plan.stats.balance_target == WindowRepairBalanceTarget::UNDERLOADED &&
          plan.stats.balance_moved == 3 && owners_of(plan.actions) == std::vector<uint32_t>({0, 1, 2}) &&
          window_repair_stats_json(plan.stats).find("\"balance_target\":\"underloaded\"") != std::string::npos,
          "the K = 8 plan balances with the underloaded rule and journals it");
    const auto plan4 = window_repair_plan(c, 4, WindowRepairPhase::IDLE, 1, 1, 80, {0, 0, 1, 2, 3, 0, 1, 2},
                                          std::vector<uint64_t>(8, 10), std::vector<uint64_t>(8, 1),
                                          std::vector<uint64_t>(8, 0), std::vector<uint32_t>(32, 1),
                                          std::vector<double>(4, 1.0), {}, {});
    check(plan4.valid && plan4.stats.balance_target == WindowRepairBalanceTarget::UNDERLOADED,
          "an opted-in underloaded target applies at K = 4 too");
    WindowRepairConfig d = f.config(WindowRepairBalanceTarget::LIGHTEST, 1.0);
    d.refine_budget = 0.0; d.balance_budget = 1.0;
    const auto plan8 = window_repair_plan(d, 8, WindowRepairPhase::IDLE, 1, 1, 100, f.membership, f.rows, work,
                                          std::vector<uint64_t>(10, 0), f.hist, f.price, {}, {});
    check(plan8.valid && plan8.stats.balance_target == WindowRepairBalanceTarget::LIGHTEST &&
          plan8.stats.balance_moved == 0 &&
          window_repair_stats_json(plan8.stats).find("\"balance_target\":\"lightest\"") != std::string::npos,
          "a lightest K = 8 plan journals lightest");
    std::printf("test_balance_target_k8 OK\n");
}

// Owner limits: K = 65 is refused by every public planner function that takes
// K (calibrate, refine, balance, plan, verdict) on inputs shaped for 65 owners,
// while the same inputs at K = 64 work, including a target mask that uses bit 63.
void test_owner_limits() {
    check(window_owners_valid(64) && !window_owners_valid(65) && !window_owners_valid(1) && kWindowMaxOwners == 64,
          "one owner limit, 2..64 (window_snapshot_wire.h)");
    WindowRepairConfig config; config.enabled = true; config.eps = 100.0; config.skew_trigger = 1.0;
    config.tau = 0.0; config.cooldown_windows = 1; config.refine_budget = 1.0; config.balance_budget = 1.0;
    config.weight = WindowRepairWeight::WNB;  // nb {0, 0} is passed
    for (const uint32_t K : {64u, 65u}) {
        const bool ok = K == 64;
        // Two nodes on owner 0; node 1 has all its neighbours on owner K - 1.
        std::vector<uint32_t> membership = {0, 0};
        std::vector<uint32_t> hist(2 * K, 0); hist[0] = 1; hist[K + K - 1] = 3;
        const std::vector<double> price(K, 1.0);
        std::vector<double> pred, W;
        WindowRepairCalibration calibration; std::string error;
        const bool calibrated = window_repair_calibrate(config, K, {2, 1}, {0, 0}, 1, membership, price, pred, W,
                                                        &calibration, &error);
        check(calibrated == ok && (ok ? W.size() == K && near(W[0], 3) : error.find("owner count") != std::string::npos),
              "calibrate: K = 64 works, K = 65 is refused");
        std::vector<double> loads(K, 0.0); loads[0] = 3.0;
        std::vector<uint32_t> m = membership, cd(2, 0); std::vector<uint8_t> mn(2, 0);
        std::vector<WindowBurstAction> a;
        const auto refine = window_repair_refine(config, K, hist, {2, 1}, price, {8, 8}, 100, 1, m, loads, cd, mn, a);
        check(ok ? (refine.moved == 1 && a.size() == 1 && a[0].node == 1 && a[0].target_owner == 63)
                 : (refine.candidates == 0 && a.empty() && m == membership), "refine: K = 65 refused");
        std::vector<double> loads2(K, 0.0); loads2[0] = 3.0;
        std::vector<uint32_t> mv2 = membership, cd2(2, 0); std::vector<uint8_t> mn2(2, 0);
        std::vector<WindowBurstAction> a2;
        const auto balance = window_repair_balance(config, K, hist, {2, 1}, price, {8, 8}, 100, 1, mv2, loads2, cd2,
                                                   mn2, a2);
        check(ok ? (balance.triggered && balance.moved >= 1) : (!balance.triggered && a2.empty() && mv2 == membership),
              "balance: K = 65 refused");
        const auto plan = window_repair_plan(config, K, WindowRepairPhase::IDLE, 1, 1, 16, membership, {8, 8}, {2, 1},
                                             {0, 0}, hist, price, {}, {});
        check(plan.valid == ok && (ok || plan.error.find("owner count") != std::string::npos), "plan: K = 65 refused");
        const auto verdict = window_repair_verdict(config, K, std::vector<double>(K, 1.0), std::vector<double>(K, 1.0),
                                                   1.0, 1.0, 90.0, 90.0);
        check(verdict.accepted == ok && (ok || !std::isfinite(verdict.peak_ratio)), "verdict: K = 65 fails closed");
    }
    // K = 64 target masks: bit 63 open moves node 1 to owner 63; bit 63 closed removes the candidate.
    const uint32_t K = 64;
    std::vector<uint32_t> hist(2 * K, 0); hist[0] = 1; hist[K + 63] = 3;
    const std::vector<double> price(K, 1.0);
    std::vector<uint64_t> open = {~uint64_t{0} ^ 1u, ~uint64_t{0} ^ 1u};
    std::vector<uint64_t> closed = open; closed[1] &= ~(uint64_t{1} << 63);
    WindowRepairConfig refine_only = config; refine_only.balance_budget = 0.0;
    const auto with_bit = window_repair_plan(refine_only, K, WindowRepairPhase::IDLE, 1, 1, 16, {0, 0}, {8, 8}, {2, 1},
                                             {0, 0}, hist, price, {}, {}, &open);
    const auto without_bit = window_repair_plan(refine_only, K, WindowRepairPhase::IDLE, 1, 1, 16, {0, 0}, {8, 8},
                                                {2, 1}, {0, 0}, hist, price, {}, {}, &closed);
    check(with_bit.valid && with_bit.actions.size() == 1 && with_bit.actions[0].target_owner == 63 &&
          without_bit.valid && without_bit.actions.empty() && without_bit.stats.refine_candidates == 0,
          "a K = 64 mask: bit 63 decides the move to owner 63");
    std::printf("test_owner_limits OK\n");
}
}  // namespace

// A later controller phase covers ids inserted since the previous one. The
// planner state (pred, cooldown_until) of the earlier phase covers the first 12
// ids of Fixture A: calibration keeps the EMA of those 12 and starts the 4 new
// ids from their first measurement (no reset of the known ids), and the carried
// cooldowns stay while the new ids are padded off cooldown. A state longer
// than the membership is refused.
void test_pad_new_nodes() {
    FixtureA f;
    WindowRepairConfig config = f.config();
    std::vector<double> pred(f.pred.begin(), f.pred.begin() + 12), W;
    for (auto& value : pred) value *= 3.0;  // an earlier phase's smoothed weights
    const std::vector<double> earlier = pred;
    WindowRepairCalibration out; std::string error;
    check(window_repair_calibrate(config, 4, f.work, f.nb_work, 10, f.membership, f.price, pred, W, &out, &error) &&
          pred.size() == 16, "a shorter pred is padded, not reset");
    for (size_t v = 0; v < 16; ++v) {
        const double expected = v < 12 ? 0.5 * f.pred[v] + 0.5 * earlier[v] : f.pred[v];
        check(near(pred[v], expected), "EMA for the known ids, the first measurement for the new ones");
    }
    std::vector<double> longer(17, 1.0);
    check(!window_repair_calibrate(config, 4, f.work, f.nb_work, 10, f.membership, f.price, longer, W, &out, &error) &&
          longer.size() == 17 && error.find("more nodes") != std::string::npos, "a longer pred is refused");
    std::vector<uint32_t> cooldown(12, 0);
    cooldown[11] = 9;  // node 11 (refine's best candidate) on cooldown until window 9
    const std::vector<double> earlier_pred(earlier);
    const auto plan = window_repair_plan(config, 4, WindowRepairPhase::ACTIVE, 3, 10, 1600, f.membership, f.row_bytes,
                                         f.work, f.nb_work, f.hist, f.price, earlier_pred, cooldown);
    check(plan.valid && plan.pred.size() == 16 && plan.cooldown_until.size() == 16 && plan.cooldown_until[11] == 9,
          "the plan pads both and keeps the carried cooldown");
    for (const auto& action : plan.actions) check(action.node != 11, "a carried cooldown still excludes its node");
    check(!window_repair_plan(config, 4, WindowRepairPhase::ACTIVE, 3, 10, 1600, f.membership, f.row_bytes, f.work,
                              f.nb_work, f.hist, f.price, earlier_pred, std::vector<uint32_t>(17, 0)).valid,
          "cooldowns longer than the membership are refused");
    std::printf("test_pad_new_nodes OK\n");
}

int main() {
    test_calibration();
    test_pad_new_nodes();
    test_weight_w();
    test_refine();
    test_balance();
    test_balance_matches_reference();
    test_balance_target_k8();
    test_plan_matches_reference_for_k();
    test_owner_limits();
    test_plan_union_and_phases();
    test_allowed_targets();
    test_verdict();
    test_caps_identity_and_token();
    test_convergence_and_guard();
    std::printf("ALL WINDOW REPAIR PLANNER TESTS PASSED\n");
    return 0;
}
