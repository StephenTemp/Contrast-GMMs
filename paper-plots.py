import matplotlib.pyplot as plt

# Ablation Plot
# -----------------

lambdas = [0, 0.25, 0.5, 0.75, 1]

cnt_gmms = [0.7, 0.72, 0.74, 0.76, 0.78]
thresholds = [0.45, 0.56, 0.67, 0.78, 0.88]
clip2pt = [0.67, 0.67, 0.67, 0.67, 0.67 ]

plt.plot(lambdas, cnt_gmms,'-ok', label="Contrast-GMM")
plt.plot(lambdas, thresholds,'-+', label="Threshold")
plt.plot(lambdas, clip2pt,'--', label="Clip2Point")
plt.title("Lambda Accuracy by Method")

plt.ylabel("Accuracy")
plt.xlabel("Lambda")
plt.legend()
plt.savefig('exp-plot.jpeg')
plt.show()