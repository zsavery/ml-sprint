import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

if __name__ == "__main__":
    print("[Start]")
    np.random.seed(2)
    x = np.random.normal(3, 1, 100)
    y = np.random.normal(159, 40, 100) / x

    # plt.scatter(x, y)
    # plt.show()

    train_x, train_y = x[:80], y[:80]
    test_x, test_y = x[80:], y[80:]

    my_model = np.poly1d(np.polyfit(train_x, train_y, 4))
    my_line = np.linspace(0, 6, 100)

    plt.scatter(train_x, train_y)
    plt.plot(my_line, my_model(my_line))
    plt.show()

    r2 = r2_score(train_y, my_model(train_x))
    print(f"Relationship: {r2}")

    r2_test = r2_score(test_y, my_model(test_x))
    print("Relationship: {}".format(r2_test))

    for i in range(7,10):
        print("[{}:{}]".format(i, my_model(i)))

    print("[Complete]")