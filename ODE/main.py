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


def calculate_difference_quotients(f, x_values, h):
    """differenzquotienten berechnen"""
    forward_diff = np.array([forward_difference_quotient(f, x, h) for x in x_values])
    backward_diff = np.array([backward_difference_quotient(f, x, h) for x in x_values])
    central_diff = np.array([central_difference_quotient(f, x, h) for x in x_values])
    exact_diff = f_prime(x_values)
    return forward_diff, backward_diff, central_diff, exact_diff


def plot_difference_quotients(x_values, results, filename):
    """differenzquotienten plotten"""
    forward_diff, backward_diff, central_diff, exact_diff = results

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
    plt.savefig(filename)
    plt.close()


def calculate_euler_errors(rhs, exact_solution, y0, t0, T, N_values):
    """fehler der impliziten und expliziten euler methoden fur gegebene N werte berechnen"""
    errors_explicit = []
    errors_implicit = []

    for N in N_values:
        t_exp, y_exp = explicit_euler(rhs, y0, t0, T, N)
        t_imp, y_imp = implicit_euler(rhs, y0, t0, T, N)
        y_exact_T = exact_solution(T)
        errors_explicit.append(abs(y_exact_T - y_exp[-1]))
        errors_implicit.append(abs(y_exact_T - y_imp[-1]))

    return errors_explicit, errors_implicit


def plot_euler_errors(N_values, errors_explicit, errors_implicit, filename):
    """plotten der euler fehler gegen schritte"""
    plt.figure()
    plt.loglog(N_values, errors_explicit, 'b-', label='explicit euler')
    plt.loglog(N_values, errors_implicit, 'r--', label='implicit euler')
    plt.xlabel('schritte (N)')
    plt.ylabel('fehler bei t=10')
    plt.title('Error vs. Number of Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def calculate_time_errors(rhs, exact_solution, y0, t0, T, N):
    """fehlerberechnung fur ein gegebenes N"""
    t_exp, y_exp = explicit_euler(rhs, y0, t0, T, N)
    t_imp, y_imp = implicit_euler(rhs, y0, t0, T, N)
    y_exact_exp = exact_solution(t_exp)
    y_exact_imp = exact_solution(t_imp)
    error_exp = np.abs(y_exact_exp - y_exp)
    error_imp = np.abs(y_exact_imp - y_imp)
    return t_exp, error_exp, t_imp, error_imp


def plot_time_errors(t_exp, error_exp, t_imp, error_imp, filename):
    """fehler uber die zeit plotten"""
    plt.figure()
    plt.semilogy(t_exp, error_exp, 'b-', label='explicit euler')
    plt.semilogy(t_imp, error_imp, 'r--', label='implicit euler')
    plt.xlabel('zeit (t)')
    plt.ylabel('fehler |y_exakt - y_numerisch|')
    plt.title('Error vs. Time (N=10,000)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def calculate_euler_solutions(rhs, exact_solution, y0, t0, T, N_values):
    """euler losungen fur verschiedene schrittbreite berechnen"""
    solutions = {
        'explicit': {'h=0.5': None, 'h=0.2': None},
        'implicit': {'h=0.5': None, 'h=0.2': None}
    }

    for N in N_values:
        t_exp, y_exp = explicit_euler(rhs, y0, t0, T, N)
        t_imp, y_imp = implicit_euler(rhs, y0, t0, T, N)
        if N == 10:
            solutions['explicit']['h=0.5'] = (t_exp, y_exp)
            solutions['implicit']['h=0.5'] = (t_imp, y_imp)
        elif N == 25:
            solutions['explicit']['h=0.2'] = (t_exp, y_exp)
            solutions['implicit']['h=0.2'] = (t_imp, y_imp)

    return solutions


def plot_euler_comparison(solutions, exact_solution, t0, T, filename):
    """plotten der euler comparison"""
    plt.figure(figsize=(12, 5))

    t_exact = np.linspace(t0, T, 100)
    y_exact = exact_solution(t_exact)

    plt.subplot(1, 2, 1)
    t_exp_h05, y_exp_h05 = solutions['explicit']['h=0.5']
    plt.plot(t_exp_h05, y_exp_h05, 'b--', label='h=0.5 (N=10)', markersize=4)
    t_exp_h02, y_exp_h02 = solutions['explicit']['h=0.2']
    plt.plot(t_exp_h02, y_exp_h02, 'g--', label='h=0.2 (N=25)', markersize=4)
    plt.plot(t_exact, y_exact, 'k-', label='Exakt')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('explizit euler-verfahren')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    t_imp_h05, y_imp_h05 = solutions['implicit']['h=0.5']
    plt.plot(t_imp_h05, y_imp_h05, 'b--', label='h=0.5 (N=10)', markersize=4)
    t_imp_h02, y_imp_h02 = solutions['implicit']['h=0.2']
    plt.plot(t_imp_h02, y_imp_h02, 'g--', label='h=0.2 (N=25)', markersize=4)
    plt.plot(t_exact, y_exact, 'k-', label='Exakt')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('implizit euler-verfahren')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    x_values = np.linspace(0, 2 * np.pi, 100)
    h = 0.2
    diff_results = calculate_difference_quotients(f, x_values, h)
    plot_difference_quotients(x_values, diff_results, 'Plots/Différence_Quotient_Comparison.pdf')

    t0, T, y0 = 0, 10, 0.25  # initialwerte 3.3b
    N_values = [100, 500, 1000, 5000, 10000]  # N werte 3.3b
    errors_exp, errors_imp = calculate_euler_errors(rhs, exakte_loesung, y0, t0, T, N_values)
    plot_euler_errors(N_values, errors_exp, errors_imp, 'Plots/Euler_Error_VS_N.pdf')

    N = 10000  # N wert 3.3b für fehler uber der zeit
    t_exp, error_exp, t_imp, error_imp = calculate_time_errors(rhs, exakte_loesung, y0, t0, T, N)
    plot_time_errors(t_exp, error_exp, t_imp, error_imp, 'Plots/Euler_Error_VS_Time.pdf')

    t0, T, y0 = 0, 5, 1  # 3.3c werte
    N_values = [10, 25]
    solutions = calculate_euler_solutions(rhs2, exakte_loesung2, y0, t0, T, N_values)
    plot_euler_comparison(solutions, exakte_loesung2, t0, T, 'Plots/Comparison_Explicit_Implicit_Euler.pdf')
