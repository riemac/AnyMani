
import torch
import numpy as np

def analyze_weights():
    print("Analyzing Jacobian Stability with different weights...")
    
    # Original Jacobian from the log (6x4)
    # Rows 0-2: Angular, Rows 3-5: Linear
    J_b_np = np.array([
        [-0.370382, -0.52345,  -0.,       -0.      ],
        [-0.,        0.826098,  1.,        1.      ],
        [ 0.928879, -0.208713, -0.,       -0.      ],
        [ 0.212818,  0.055096,  0.146474,  0.132792],
        [-0.501878,  0.19871,   0.,        0.      ],
        [ 0.084859,  0.648327,  0.582561,  0.549155]
    ])
    
    J_b = torch.tensor(J_b_np, dtype=torch.float32)
    
    # Damping factor from config
    damping = 0.01
    lambda_sq = damping ** 2
    
    weights_to_test = [1.0, 5.0, 10.0, 20.0]
    
    print(f"{'Weight':<10} | {'Cond No':<10} | {'Max SV':<10} | {'Min SV':<10} | {'Cond(JJT+lam)':<15} | {'Cholesky?':<10}")
    print("-" * 80)
    
    for w_lin in weights_to_test:
        # Construct Weight Matrix W_x
        # Angular weights = 1.0, Linear weights = w_lin
        w_diag = torch.tensor([1.0, 1.0, 1.0, w_lin, w_lin, w_lin], dtype=torch.float32)
        W_x = torch.diag(w_diag)
        
        # Weighted Jacobian: J_w = W_x @ J_b
        # Assuming W_q is Identity for simplicity (as per log)
        J_w = W_x @ J_b
        
        # SVD Analysis
        try:
            U, S, Vh = torch.linalg.svd(J_w)
            max_sv = S[0].item()
            min_sv = S[-1].item()
            cond_no = max_sv / min_sv if min_sv > 1e-9 else float('inf')
        except Exception as e:
            max_sv, min_sv, cond_no = -1, -1, -1
            
        # DLS Matrix: A = J_w @ J_w.T + lambda^2 * I
        # Note: For under-actuated (6x4), DLS usually formulates in joint space or uses pseudo-inverse.
        # The formula in se3wdlsAction docstring is:
        # theta_dot = W_q^-1 J_t^T (J_t J_t^T + lambda^2 I)^-1 W_x V
        # So we invert (J_t J_t^T + lambda^2 I) which is 6x6.
        # Wait, J_t is 6x4. J_t J_t^T is 6x6. Rank is at most 4.
        # So J_t J_t^T is singular (rank 4 < 6).
        # Adding lambda^2 I makes it full rank.
        
        A = J_w @ J_w.T + lambda_sq * torch.eye(6, dtype=torch.float32)
        
        # Condition number of the matrix to be inverted
        try:
            U_A, S_A, Vh_A = torch.linalg.svd(A)
            cond_A = S_A[0] / S_A[-1]
        except:
            cond_A = torch.tensor(float('nan'))

        # Cholesky Check
        try:
            L = torch.linalg.cholesky(A)
            cholesky_ok = "OK"
        except RuntimeError:
            cholesky_ok = "FAIL"
            
        print(f"{w_lin:<10.1f} | {cond_no:<10.2f} | {max_sv:<10.4f} | {min_sv:<10.4f} | {cond_A.item():<15.2e} | {cholesky_ok:<10}")

if __name__ == "__main__":
    analyze_weights()
