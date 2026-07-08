#include <cassert>

#include <torchless/tensor.h>

void test_factories_1D(Device dev) {
    Tensor t({1, 2}, dev);
    assert(t.get({0}) == 1);
    assert(t.get({1}) == 2);

    t = Tensor::filled({5}, 2, dev);
    assert(t.get({0}) == 2);
    assert(t.get({4}) == 2);

    t = Tensor::zeros({5}, dev);
    assert(t.get({0}) == 0);
    assert(t.get({4}) == 0);

    t = Tensor::ones({5}, dev);
    assert(t.get({0}) == 1);
    assert(t.get({4}) == 1);

    int seed = 42;
    Tensor t1 = Tensor::rand({3}, std::mt19937(seed), dev);
    Tensor t2 = Tensor::rand({3}, std::mt19937(seed), dev);

    for (size_t i = 0; i < 3; ++i) {
        double val = t1.get({i});
        assert(val == t2.get({i}));

        if (!(0 <= val && val < 1)) {
            std::cout << i << ": " << val << std::endl;
        }
        assert(0 <= val && val < 1);
    }
}

void test_factories_2D(Device dev) {
    Tensor t({{1, 2}, {3, 4}}, dev);

    assert(t.get({0, 0}) == 1);
    assert(t.get({0, 1}) == 2);
    assert(t.get({1, 1}) == 4);

    t = Tensor::filled({5, 2}, 2, dev);
    assert(t.get({3, 0}) == 2);
    assert(t.get({4, 1}) == 2);

    t = Tensor::zeros({5, 2}, dev);
    assert(t.get({3, 0}) == 0);
    assert(t.get({4, 1}) == 0);

    t = Tensor::ones({5, 2}, dev);
    assert(t.get({3, 0}) == 1);
    assert(t.get({4, 1}) == 1);

    int seed = 42;
    Tensor t1 = Tensor::rand({3, 3}, std::mt19937(seed), dev);
    Tensor t2 = Tensor::rand({3, 3}, std::mt19937(seed), dev);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double val = t1.get({i, j});
            assert(val == t2.get({i, j}));
            assert(0 <= val && val < 1);
        }
    }
}
