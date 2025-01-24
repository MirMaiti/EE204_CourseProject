import numpy as np
from sympy import symbols, Matrix, eye, zeros
from sympy.tensor.array.expressions import OneArray

s, t = symbols('s t') 

def get_branches_order(tree_branches, reversed_branch_name):
    branches_order = tree_branches.copy()
    tree_branches.sort()
    for key, value in reversed_branch_name.items():
        if key not in tree_branches:
            branches_order.append(key)
    return branches_order


def get_matrix_a(tree_branches, reversed_branch_name, nodes, branches):
    branches_order = get_branches_order(tree_branches, reversed_branch_name)
    matrix_a = [[0 for _ in range(branches)] for __ in range(nodes)]
    for branch in range(branches):
        (key, value) = reversed_branch_name[branches_order[branch]]
        matrix_a[key][branch], matrix_a[value][branch] = 1, -1
    return Matrix(matrix_a)


def get_matrix_a_tree(tree_branches, reversed_branch_name, nodes, branches):
    matrix_a = get_matrix_a(tree_branches, reversed_branch_name, nodes, branches)
    matrix_a_tree = matrix_a[:nodes - 1, :len(tree_branches)]
    return matrix_a_tree


def get_matrix_a_link(tree_branches, reversed_branch_name, nodes, branches):
    matrix_a = get_matrix_a(tree_branches, reversed_branch_name, nodes, branches)
    matrix_a_link = matrix_a[:nodes - 1, len(tree_branches):]
    return matrix_a_link


def get_matrix_c_link(matrix_a_tree, matrix_a_link):
    matrix_a_tree_inverse = matrix_a_tree.inv()
    # aTreeInverse = aTree ^ -1
    matrix_c_link = matrix_a_tree_inverse * matrix_a_link
    return matrix_c_link


def get_matrix_b_tree(matrix_a_tree, matrix_a_link):
    matrix_c_link = get_matrix_c_link(matrix_a_tree, matrix_a_link)
    matrix_b_tree = -matrix_c_link.transpose()
    matrix_b_tree[matrix_b_tree == -0] = 0  # avoiding '-0'
    return matrix_b_tree


def get_matrix_c(matrix_a_tree, matrix_a_link):
    matrix_c_link = get_matrix_c_link(matrix_a_tree, matrix_a_link)
    matrix_c_tree = eye(matrix_a_tree.rows)
    return matrix_c_tree.row_join(matrix_c_link)


def get_matrix_b(matrix_a_tree, matrix_a_link):
    matrix_b_tree = get_matrix_b_tree(matrix_a_tree, matrix_a_link)
    matrix_b_link = eye(matrix_b_tree.rows)
    return matrix_b_tree.row_join(matrix_b_link)


def get_matrix_impedance(resistors, capacitors, inductors, s):
    num_branches = len(resistors)
    impedance_matrix = zeros(num_branches, num_branches)
    for i in range(num_branches):
        if resistors[i] is not None:
            impedance_matrix[i, i] += resistors[i]
        if inductors[i] is not None:
            impedance_matrix[i, i] += s * inductors[i]
        if capacitors[i] is not None and capacitors[i] != 0:
            impedance_matrix[i, i] += 1 / (s * capacitors[i])
    return impedance_matrix


def get_matrix_current_source(current_sources):
    matrix_current_source = [[current_sources[i]] for i in range(len(current_sources))]
    return Matrix(matrix_current_source)


def get_matrix_voltage_source(voltage_sources):
    matrix_voltage_source = [[voltage_sources[i]] for i in range(len(voltage_sources))]
    return Matrix(matrix_voltage_source)


def get_matrix_i_loop(matrix_b, matrix_impedance, matrix_current_source, matrix_voltage_source):
    right_side = (matrix_b * matrix_voltage_source) + (matrix_b * matrix_impedance * matrix_current_source)
    left_side = matrix_b * matrix_impedance * matrix_b.transpose()
    left_side_inverse = left_side.inv()    
    matrix_i_loop = (left_side_inverse * right_side)
    return matrix_i_loop


def get_matrix_j_branch(matrix_b, matrix_i_loop):
    matrix_j_branch = matrix_b.transpose() * matrix_i_loop
    return matrix_j_branch


def get_matrix_v_branch(matrix_j_branch, matrix_impedance, matrix_current_source, matrix_voltage_source):
    matrix_v_branch = matrix_impedance * (matrix_j_branch.transpose() + matrix_current_source) - matrix_voltage_source
    return matrix_v_branch.transpose()