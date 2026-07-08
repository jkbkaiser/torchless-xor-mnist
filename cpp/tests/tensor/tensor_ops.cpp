#include <cassert>

#include <torchless/tensor.h>
#include <torchless/utils.h>

void test_unary_ops(Device dev) {
    bool ok = true;

    Tensor t({{1, 2}, {3, 4}}, dev);
    Tensor logged = t.log();
    Tensor logged_cpu = logged.to(Device::CPU);
    std::vector<std::vector<float>> expected = {{std::log(1.f), std::log(2.f)},
                                                {std::log(3.f), std::log(4.f)}};
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float val = logged_cpu.get({i, j});
            if (std::abs(val - expected[i][j]) > 1e-5f) {
                ok = false;
            }
        }
    }
    assert(ok);

    Tensor exped = t.exp();
    Tensor exped_cpu = exped.to(Device::CPU);
    expected = {{std::exp(1.f), std::exp(2.f)}, {std::exp(3.f), std::exp(4.f)}};

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float val = exped_cpu.get({i, j});
            if (std::abs(val - expected[i][j]) > 1e-5f) {
                ok = false;
            }
        }
    }
    assert(ok);

    Tensor t1(std::vector<float>{4}, dev);
    Tensor t2 = t1.exp().log().to(Device::CPU);
    assert(std::abs(t2.get({0}) - 4.f) < 1e-5f);

    t2 = t1.log().exp().to(Device::CPU);
    assert(std::abs(t2.get({0}) - 4.f) < 1e-5f);

    t2 = -t1;
    assert(t2.get({0}) == -t1.get({0}));
    t2 = -t2;
    assert(t2.get({0}) == t1.get({0}));
}

void test_tensor_scalar_ops(Device dev) {
    Tensor t1({{1, 2}, {3, 4}}, dev);
    Tensor t2 = t1 + 3;
    assert(t2.get({0, 0}) == 4);
    assert(t2.get({1, 1}) == 7);

    Tensor t3 = t2 - 2;
    assert(t3.get({0, 0}) == 2);
    assert(t3.get({1, 1}) == 5);

    Tensor t4 = t1 * 2;
    assert(t4.get({0, 0}) == 2);
    assert(t4.get({1, 1}) == 8);

    Tensor t5 = t4 / 2;
    assert(t5.get({0, 0}) == 1);
    assert(t5.get({1, 1}) == 4);

    Tensor t6 = 3 + t1;
    assert(t6.get({0, 0}) == 4);
    assert(t6.get({1, 1}) == 7);

    Tensor t7 = 5 - t1;
    assert(t7.get({0, 0}) == 4);
    assert(t7.get({1, 1}) == 1);

    Tensor t8 = 2 * t1;
    assert(t8.get({0, 0}) == 2);
    assert(t8.get({1, 1}) == 8);

    Tensor t9 = 12 / t1;
    assert(t9.get({0, 0}) == 12);
    assert(t9.get({1, 1}) == 3);
}

void test_tensor_tensor_ops(Device dev) {
    Tensor t1({{1, 2}, {3, 4}}, dev);
    Tensor t2({{1, 2}, {3, 5}}, dev);
    Tensor t3 = t1 == t2;
    assert(t3.get({0, 0}) == 1);
    assert(t3.get({1, 1}) == 0);

    Tensor t4({{1, 2}, {3, 4}, {5, 6}}, dev);
    Tensor t5({1, 2}, dev);
    std::cout << "\nshape: " << t5.shape() << std::endl;
    Tensor t6 = t4 + t5;

    std::cout << "t6\n" << t6 << std::endl;
    assert(t6.get({0, 0}) == 2);
    assert(t6.get({1, 0}) == 4);
    assert(t6.get({2, 1}) == 8);
}
