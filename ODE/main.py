# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from ode import (forward_difference_quotient, backward_difference_quotient, central_difference_quotient,
                 explicit_euler, implicit_euler)


def f(x):
    return np.sin(x)  # 3.3a funktion


def f_prime(x):
    return np.cos(x)


def rhs(t, y):
    return 0.5 * (1 - y / 8) * y  # erste ODE 3.3b


def exakte_loesung(t):
    return 2 / (0.25 + 7.75 * np.exp(-0.5 * t))


def rhs2(t, y):
    return -4 * (y - 2)  # zweite ODE 3.3c


def exakte_loesung2(t):
    return 2 - np.exp(-4 * t)


if __name__ == '__main__':
    x_values = np.linspace(0, 2 * np.pi, 100)
    h = 0.2
    forward_diff = np.array([forward_difference_quotient(f, x, h) for x in x_values])
    backward_diff = np.array([backward_difference_quotient(f, x, h) for x in x_values])
    central_diff = np.array([central_difference_quotient(f, x, h) for x in x_values])
    exact_diff = f_prime(x_values)
    plt.figure()
    plt.plot(x_values, exact_diff, 'k-', label='ableitung')
    plt.plot(x_values, forward_diff, 'r--', label='vorwärts-differenzenquotient', markersize=4)
    plt.plot(x_values, backward_diff, 'b--', label='rückwärts-differenzenquotient', markersize=4)
    plt.plot(x_values, central_diff, 'g--', label='zentraler differenzenquotient', markersize=4)
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('Vergleich der Differenzenquotienten für f(x) = sin(x)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/Différence_Quotient_Comparison.pdf')
    plt.close()
    t0, T, y0 = 0, 10, 0.25 #initialwerte 3.3b
    N_values = [100, 500, 1000, 5000, 10000] #N werte 3.3b
    errors_explicit = []
    errors_implicit = []
    for N in N_values:
        t_exp, y_exp = explicit_euler(rhs, y0, t0, T, N)
        t_imp, y_imp = implicit_euler(rhs, y0, t0, T, N)
        y_exact_T = exakte_loesung(T)
        errors_explicit.append(abs(y_exact_T - y_exp[-1]))
        errors_implicit.append(abs(y_exact_T - y_imp[-1]))
    plt.figure()
    plt.loglog(N_values, errors_explicit, 'b-', label='explicit euler')
    plt.loglog(N_values, errors_implicit, 'r--', label='implicit euler')
    plt.xlabel('schritte (N)')
    plt.ylabel('fehler bei t=10')
    plt.title('Error vs. Number of Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/Euler_Error_VS_N.pdf')
    plt.close()
    N = 10000 #N werte 3.3b für fehler uber der zeit
    t_exp, y_exp = explicit_euler(rhs, y0, t0, T, N)
    t_imp, y_imp = implicit_euler(rhs, y0, t0, T, N)
    y_exact_exp = exakte_loesung(t_exp)
    y_exact_imp = exakte_loesung(t_imp)
    error_exp = np.abs(y_exact_exp - y_exp)
    error_imp = np.abs(y_exact_imp - y_imp)
    plt.figure()
    plt.semilogy(t_exp, error_exp, 'b-', label='explicit euler')
    plt.semilogy(t_imp, error_imp, 'r--', label='implicit euler')
    plt.xlabel('zeit (t)')
    plt.ylabel('fehler |y_exakt - y_numerisch|')
    plt.title('Error vs. Time (N=10,000)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/Euler_Error_VS_Time.pdf')
    plt.close()
    t0, T, y0 = 0, 5, 1
    N_values = [10, 25]
    losungen = {
        'explicit': {'h=0.5': None, 'h=0.2': None},
        'implicit': {'h=0.5': None, 'h=0.2': None}
    }
    for N in N_values:
        t_exp, y_exp = explicit_euler(rhs2, y0, t0, T, N)
        t_imp, y_imp = implicit_euler(rhs2, y0, t0, T, N)
        if N == 10:
            losungen['explicit']['h=0.5'] = (t_exp, y_exp)
            losungen['implicit']['h=0.5'] = (t_imp, y_imp)
        elif N == 25:
            losungen['explicit']['h=0.2'] = (t_exp, y_exp)
            losungen['implicit']['h=0.2'] = (t_imp, y_imp)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    t_exact = np.linspace(t0, T, 100)
    y_exact = exakte_loesung2(t_exact)
    t_exp_h05, y_exp_h05 = losungen['explicit']['h=0.5']
    plt.plot(t_exp_h05, y_exp_h05, 'b--', label='h=0.5 (N=10)', markersize=4)
    t_exp_h02, y_exp_h02 = losungen['explicit']['h=0.2']
    plt.plot(t_exp_h02, y_exp_h02, 'g--', label='h=0.2 (N=25)', markersize=4)
    plt.plot(t_exact, y_exact, 'k-', label='Exakt')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('explizit euler-verfahren')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    t_imp_h05, y_imp_h05 = losungen['implicit']['h=0.5']
    plt.plot(t_imp_h05, y_imp_h05, 'b--', label='h=0.5 (N=10)', markersize=4)
    t_imp_h02, y_imp_h02 = losungen['implicit']['h=0.2']
    plt.plot(t_imp_h02, y_imp_h02, 'g--', label='h=0.2 (N=25)', markersize=4)
    plt.plot(t_exact, y_exact, 'k-', label='Exakt')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('implizit euler-verfahren')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plots/Comparison_Explicit_Implicit_Euler.pdf')
    plt.close()
