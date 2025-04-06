#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics. Here are relations between temperature and water vapor pressure.
It is based on the Guide to Meteorological Instruments and Methods of Observation (CIMO Guide) (WMO, 2008)
"""

import numpy as np
# import autograd.numpy as np  # mandatory to run tests

# vapor saturation pressure
"""
#   Bolton 1980
def	ew( TK ):
    t = TK - 273.15
    if t > 100:
        return ew(373.15);
    if t < -200:
        return ew(73.15);
#    if t < -64.8925:
#        return 0.01;
    return (6.112*np.exp(17.67*t/(243.5+t)));
"""

#
# def ew(TK: float) -> float:
#     """
#     :param TK: temperature (in Kelvin)
#     :return: Mixture's saturation vapour pressure (in hPa)
#     """
#     t = TK - 273.15
#     if t > 100:
#         t = 100
#     if t < -200:
#         t = -200
#     res = 6.112 * np.exp(17.62 * t / (243.12 + t))
#     return res
#
#
# def dew_dT(TK: float) -> float:
#     """
#     :param TK: temperature (in Kelvin)
#     :return: Derivative of mixture's saturation vapour pressure with respect to TK
#     """
#     t = TK - 273.15
#     if t > 100:
#         return 0
#     if t < -200:
#         return 0
#     d = 1 / (243.12 + t)
#     res = 26182.4291328 * d * d * np.exp(17.62 * t * d)  # 26182.4291328 == 6.112*17.62*243.12
#     return res
#
#
# def dlogew_dT(TK: float) -> float:
#     """
#     :param TK: temperature (in Kelvin)
#     :return: Logarithmic derivative of mixture's saturation vapour pressure with respect to TK
#     """
#     t = TK - 273.15
#     if t > 100:
#         return 0
#     if t < -200:
#         return 0
#     d = 1 / (243.12 + t)
#     res = 17.62 * 243.12 * d * d  # 26182.4291328 == 6.112*17.62*243.12
#     return res
#
#
# ew_vec = np.vectorize(ew)
# dew_dT_vec = np.vectorize(dew_dT)
# dlogew_dT_vec = np.vectorize(dlogew_dT)
#
#
# def qsat(p_hPa: float, T_K: float) -> float:
#     """
#     The specific humidity q = m_water / m_air.
#     see https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
#
#     Note:
#     The relative humidity r = m_water / m_dry, where m_dry = (m_air - m_water).
#     The relation between r and q is q = r / (1 + r). It's almost the same because r is always very small (< 0.04).
#     See Humidity Variable p. 74  of book Atmospheric Thermodynamics by Iribarne and Godson. (ask Clement).
#     See my Notion page (Q. Pikeroen).
#
#     :param p_hPa: pressure (in hPa)
#     :param T_K: temperature (in Kelvin)
#     :return: The mixing ration r = m_water / (m_air - m_water)
#     """
#     e = ew(T_K)
#     return 0.622 * e / (p_hPa - 0.378 * e)
#
# def dqsat_dT(p_hPa: float, T_K: float) -> float:
#     """
#     :param p_hPa: pressure (in hPa)
#     :param T_K: temperature (in Kelvin)
#     :return: Derivative of the mixing ratio, with respect to temperature
#     """
#     e = ew(T_K)
#     de = dew_dT(T_K)
#     deno = 1 / (p_hPa - 0.378 * e)
#     return 0.622 * p_hPa * de * deno * deno
#
#
# qsat_vec = np.vectorize(qsat)
# dqsat_dT_vec = np.vectorize(dqsat_dT)

# TEST trying not to use vectorize
def ew_vec(TK: np.ndarray) -> np.ndarray:
    """
    :param TK: temperature (in Kelvin)
    :return: Mixture's saturation vapour pressure (in hPa)
    """
    t = TK - 273.15

    for idx, t_idx in enumerate(t):
        # print("idx", idx, "t", t)
        if t_idx > 100:
            t[idx] = 100
        if t_idx < -200:
            t[idx] = -200
    res = 6.112 * np.exp(17.62 * t / (243.12 + t))
    return res

def dew_dT_vec(TK: np.ndarray) -> np.ndarray:
    """
    :param TK: temperature (in Kelvin)
    :return: Derivative of mixture's saturation vapour pressure with respect to TK
    """
    res = np.zeros_like(TK)
    t = TK - 273.15
    for idx, t_idx in enumerate(t):
        if t_idx > 100:
            res[idx] = 0
        elif t_idx < -200:
            res[idx] = 0
        else:
            d = 1 / (243.12 + t_idx)
            res[idx] = 26182.4291328 * d * d * np.exp(17.62 * t_idx * d)  # 26182.4291328 == 6.112*17.62*243.12
    return res

def qsat_vec(p_hPa: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    """
    The specific humidity q = m_water / m_air.
    see https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html

    Note:
    The relative humidity r = m_water / m_dry, where m_dry = (m_air - m_water).
    r = 0.622 * e / (p_hPa - e)
    The relation between r and q is q = r / (1 + r). It's almost the same because r is always very small (< 0.04).
    See Humidity Variable p. 74  of book Atmospheric Thermodynamics by Iribarne and Godson. (ask Clement).
    See my Notion page (Q. Pikeroen).

    :param p_hPa: pressure (in hPa)
    :param T_K: temperature (in Kelvin)
    :return: The mixing ration r = m_water / (m_air - m_water)
    """
    e = ew_vec(T_K)
    return 0.622 * e / (p_hPa - 0.378 * e)


def dqsat_dT_vec(p_hPa: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    """
    :param p_hPa: pressure (in hPa)
    :param T_K: temperature (in Kelvin)
    :return: Derivative of the mixing ratio, with respect to temperature
    """
    e = ew_vec(T_K)
    de = dew_dT_vec(T_K)
    deno = 1 / (p_hPa - 0.378 * e)
    return 0.622 * p_hPa * de * deno * deno


