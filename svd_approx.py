import numpy as np
import matplotlib.pyplot as plt


def svd_approx(A : np.ndarray, k : int) -> np.ndarray:
    """ Creates an approximate matrix of rank k from the input matrix A (not necessarily square)
    """
    U, S, V = np.linalg.svd(A)
    # Create a diagonal matrix with the first k singular values
    S = np.diag(S[:k])
    # Create a matrix with the first k columns of U
    U = U[:, :k]
    # Create a matrix with the first k rows of V
    V = V[:k, :]
    # Calculate the approximate matrix
    A_appr = U @ (S @ V)
    return A_appr


def svd_approx_error(A : np.ndarray, A_appr : np.ndarray) -> dict:
    """ Calculates the error between the original matrix and the approximate matrix
    using matrix norm, and trace
    """
    error = {}
    error["frob"] = np.linalg.norm(A - A_appr, ord="fro").round(5)
    error["L1"] = np.linalg.norm(A - A_appr, ord=1).round(5)
    error["L2"] = np.linalg.norm(A - A_appr, ord=2).round(5)
    error["trace"] = np.trace(np.abs(A - A_appr)).round(5)
    return error


if __name__ == "__main__":
    #A = np.random.rand(1000,50)
    A = np.array([[3,2,5,1,0,2,4,3,1,0],
                  [0,1,3,4,1,2,0,0,2,4],
                  [2,1,3,4,2,5,0,0,3,0],
                  [0,4,2,3,7,0,2,5,2,1],
                  [3,0,0,2,4,1,0,3,1,0],
                  [2,0,0,0,5,2,0,2,0,4],
                  [3,4,0,2,0,0,2,4,5,0],
                  [2,5,0,0,3,2,5,0,0,0],
                  [3,1,3,4,0,0,0,0,3,2],
                  [0,0,0,3,4,0,0,2,3,0],
                  [3,0,0,4,0,5,2,0,0,1],
                  [2,2,0,3,0,4,0,0,2,3]]
                  )
    errs = []
    ranks = list(range(1,min(A.shape)+1))
    for k in ranks:
        A_approx = svd_approx(A, k)
        err = svd_approx_error(A, A_approx)
        print(f"Rank {k} {A_approx.shape} error: {err}")
        errs.append(err)
    fig, ax = plt.subplots()
    ax.plot(ranks, [err["frob"] for err in errs], label="Frobenius")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True)
    ax.set_title("SVD Approximation Error (Frobenius norm)")

    fig1, ax1 = plt.subplots()
    ax1.plot(ranks, [err["trace"] for err in errs], label="trace")
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("SVD Approximation Error (Trace)")

    plt.show()