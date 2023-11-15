import streamlit as st 
import numpy
import csv
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

def discrete_forrier_transform(N, fs, signal):
    
    #Initial Array
    X_real = np.zeros(10000)
    X_imj = np.zeros(10000)
    MagDFT = np.zeros(10000)

    #DFT
    for k in range (N):
        for n in range (N):
            X_real[k] += (signal[n])*np.cos(2*np.pi*k*n/N)
            X_imj[k] -= (signal[n])*np.sin(2*np.pi*k*n/N)

    for k in range (N):
        MagDFT[k] = np.sqrt(np.square(X_real[k])+np.square(X_imj[k]))
    return MagDFT

def movingAverage(signal, window):
    sum = 0
    mAver_forward = []
    mAver_backward = []
    k = int((window - 1) / 2)

    # Forward Moving Average
    for i in np.arange(k, len(signal) - k):
        for ii in np.arange(i - k, i + k + 1):
            sum = sum + signal[ii]
        mAver_forward.append(sum / window)
        sum = 0

    zeros = [0] * k
    mAver_forward = zeros + mAver_forward + zeros

    # Backward Moving Average
    signal = mAver_forward[::-1]
    for i in np.arange(k, len(signal) - k):
        for ii in np.arange(i - k, i + k + 1):
            sum = sum + signal[ii]
        mAver_backward.append(sum / window)
        sum = 0

    mAver_backward = mAver_backward[::-1]
    zeros = [0] * k
    mAver_backward = zeros + mAver_backward + zeros

    return mAver_backward

def butterworth_lowpass_filter(signal, cutoff_frequency, sampling_period, orde):
    y = np.zeros(len(signal)) 
    omega_c = 2 * np.pi * cutoff_frequency
    omega_c_squared = omega_c*omega_c
    sampling_period_squared = sampling_period*sampling_period
    if orde == 1:
        for n in range(len(signal)):
            if n == 0:
                y[n] = (omega_c * signal[n]) / ((2 / sampling_period) + omega_c)
            else:
                y[n] = (((2 / sampling_period) - omega_c) * y[n-1] + omega_c * signal[n] + omega_c * signal[n-1]) / ((2 / sampling_period) + omega_c)
    elif orde == 2:
        y[0] = (omega_c * signal[0]) / ((2 / sampling_period) + omega_c)
        y[1] = (((2 / sampling_period) - omega_c) * y[0] + omega_c * signal[1] + omega_c * signal[0]) / ((2 / sampling_period) + omega_c)
        for n in range(2, len(signal)):
            y[n] = (((8/sampling_period_squared)-2*omega_c_squared) * y[n-1]
                - ((4/sampling_period_squared) - (2 * np.sqrt(2) * omega_c / sampling_period) + omega_c_squared) * y[n-2]
                + omega_c_squared * signal[n]
                + 2 * omega_c_squared * signal[n-1]
                + omega_c_squared * signal[n-2]) / ((4/sampling_period_squared) + (2 * np.sqrt(2) * omega_c / sampling_period) + omega_c_squared)

    return y

def butterworth_highpass_filter(signal, cutoff_frequency, sampling_period, orde):
    y = np.zeros(len(signal))  # Initialize the output signal
    omega_c = 2 * np.pi * cutoff_frequency
    omega_c_squared = omega_c*omega_c
    sampling_period_squared = sampling_period*sampling_period

    if orde == 1:
        for n in range(len(signal)):
            if n == 0:
                y[n] = (omega_c * signal[n]) / ((2 / sampling_period) + omega_c)
            else:
                y[n] = (((2 / sampling_period) - omega_c) * y[n-1] + (2 / sampling_period) * signal[n] + (2 / sampling_period) * signal[n-1]) / ((2 / sampling_period) + omega_c)

    elif orde == 2:
        y[0] = (omega_c * signal[0]) / ((2 / sampling_period) + omega_c)
        y[1] = (((2 / sampling_period) - omega_c) * y[0] + (2 / sampling_period) * signal[1] + (2 / sampling_period) * signal[0]) / ((2 / sampling_period) + omega_c)

        for n in range(2, (len(signal))):
            y[n] = ((4/sampling_period_squared)*signal[n] - (8/sampling_period_squared)*signal[n-1] + (4/sampling_period_squared)*signal[n-2] - (2*omega_c-(8/sampling_period_squared))*y[n-1] - (omega_c-(2*math.sqrt(2)*omega_c/sampling_period)+(4/sampling_period_squared))*y[n-2])/(omega_c + 2*math.sqrt(2)*omega_c/sampling_period + (4/sampling_period_squared))    
    i=0
    temp=y
    while i <= len(y):
        if i < 10:
            continue
        else:
            y[i]= temp[i-10]
        i+=1

    return y

def segmentation_p_qrs_t(signal_pre, N):
    signal_P=[]
    for i in range(N):
        if i>=1300 and i<=1550:
            signal_P.append(signal_pre[i])
        else:
            signal_P.append(0)

    signal_QRS=[]
    for i in range(N):
        if i>=1550 and i<=1680:
            signal_QRS.append(signal_pre[i])
        else:
            signal_QRS.append(0)

    signal_T=[]
    for i in range(N):
        if i>=1680 and i<=2000:
            signal_T.append(signal_pre[i])
        else:
            signal_T.append(0)
    
    return signal_P, signal_QRS, signal_T

def create_plotly_figure(title, xaxis, yaxis, time, signal, name, mode):
    # Create a Plotly figure
    fig = go.Figure()
    color= ['blue', 'red', 'green']
    for i in range(len(signal)):
        fig.add_trace(go.Scatter(x=time, y=signal[i], mode=mode, name=name[i], line=dict(color=color[i])))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        width=800,
        height=500,
        xaxis=dict(showline=True, showgrid=True),
        yaxis=dict(showline=True, showgrid=True)
    )

    st.plotly_chart(fig)

def count_peaks_qrs_width(signal, signal_threshold, fs):
    peaks = 0
    for i in range(0, len(signal_threshold) - 1):
        if signal_threshold[i] > signal_threshold[i - 1]:
            peaks += 1
        elif signal_threshold[i] == signal_threshold[i-1]:
            continue

    #print("ECG Cycle:", peaks, "beats in", len(signal)/fs, "second")

    durasi_dalam_menit = (len(signal)/fs) / 60
    # count bpm
    bpm = peaks / durasi_dalam_menit
    #print("BPM (Beats Per Minute):", bpm)

    # Count QRS Width 
    qrs_widths = []
    j = 0
    p=0
    i=0
    while (i<len(signal)):
        if signal_threshold[i] != 0:
            while (i < (len(signal_threshold) - 1)) and signal_threshold[i + 1] != 0:
                j += 1
                i += 1
            qrs_widths.append(j)
        j = 0
        i+=1

    avarage_qrs_widths= 0
    for i in range(len(qrs_widths)):
        avarage_qrs_widths+= qrs_widths[i]

    avarage_qrs_widths = (avarage_qrs_widths/(len(qrs_widths)))/fs
    return peaks, bpm, qrs_widths, avarage_qrs_widths


st.title("QRS Detection with IIR Butterworth Fillter")

#OPEN DATASET 
time = []
signal = []

with open('data.txt') as file:
    lines = csv.reader(file, delimiter='\t')
    for row in lines:
        values = row[0].split()
        if len(values) == 2:
            time_value = int(values[0])
            signal_value = float(values[1])
            time.append(time_value)
            signal.append(signal_value-1.25)
param = st.sidebar.selectbox("Choose Mode", ["Butterworth in Time Domain", "Frequency Domain Analysis"])
if param == "Butterworth in Time Domain":
    st.subheader("Original Signal from Dataset")
    create_plotly_figure('Original Signal','Sequence (n)','Amplitude (mV)',time, [signal], ['Original Signal'],'lines')
    st.sidebar.title("Variable of Filter")
    fs = st.sidebar.number_input("Frequency Sampling", value=1000) 

    #Preprocessing Signal 
    st.sidebar.subheader("Preprocessing")
    mav_window_pre=st.sidebar.number_input("MAV Window Preprocessing", value=20)

    st.sidebar.subheader("LPF Butterworth")
    cutoff_frequency_lpf = st.sidebar.number_input("Cutoff Frequency LPF", value=12)
    orde_LPF = st.sidebar.selectbox("Orde LPF Filter",[1, 2], index=1)
    sampling_period = 1/fs

    st.sidebar.subheader("HPF Butterworth")
    cutoff_frequency_hpf = st.sidebar.number_input("Cutoff Frequency HPF", value=12)
    orde_HPF = st.sidebar.selectbox("Orde HPF Filter",[1, 2], index=1)

    st.sidebar.subheader("Squering and MAV")
    mav_window=st.sidebar.number_input("MAV Window", value=30)

    st.sidebar.subheader("Thresholding MAV")
    threshold=st.sidebar.number_input("Threshold", value=0.15)

    if st.sidebar.button("Start Filtering"):
        st.subheader("Preprocessing Signal with MAV Filter")
        signal = movingAverage(signal, mav_window_pre)
        create_plotly_figure('Preprocessing Signal','Sequence (n)','Amplitude (mV)',time, [signal], ['Preprocessing Signal'],'lines')
        
        st.subheader("LPF Butterworth")
        filtered_signal = butterworth_lowpass_filter(signal, cutoff_frequency_lpf, sampling_period, orde_LPF)
        create_plotly_figure('Output LPF Filter','Sequence (n)','Amplitude (mV)',time, [signal, filtered_signal], ['Preprocessing Signal', 'Output LPF Filter'],'lines')

        st.subheader("HPF Butterworth")
        filtered_signal_HPF = butterworth_highpass_filter(filtered_signal, cutoff_frequency_hpf, sampling_period, orde_HPF)
        create_plotly_figure('Output HPF Filter','Sequence (n)','Amplitude (mV)',time, [filtered_signal, filtered_signal_HPF], ['Output LPF Filter', 'Output HPF Filter'],'lines')

        st.subheader("Squering and MAV")
        signal_sqrt = []
        for i in range (len(signal)):
            signal_sqrt.append(filtered_signal_HPF[i]*filtered_signal_HPF[i])
        signal_mav = movingAverage(signal_sqrt, mav_window)
        create_plotly_figure('Squering and MAV','Sequence (n)','Amplitude (mV)',time, [signal_sqrt, signal_mav], ['Squering Signal', 'MAV Signal'],'lines')

        st.subheader("Thresholding MAV")
        signal_threshold = []
        for i in range (len(signal)):
            if signal_mav[i]>threshold * np.max(signal_mav):
                signal_threshold.append(np.max(signal))
            else:
                signal_threshold.append(0)
        create_plotly_figure('Squering and MAV','Sequence (n)','Amplitude (mV)',time, [signal, signal_threshold], ['Signal', 'Signal Threshold'],'lines')

        st.subheader("Heart Rate & QRS Width")
        peaks, bpm, qrs_widths, avarage_qrs_widths = count_peaks_qrs_width(signal, signal_threshold, fs)
        st.write(f"Jumlah peak dalam dataset :{peaks} peak")
        st.write(f"Heart Rate                : {bpm} beat per minute")
        st.write("QRS Width")
        qrs_widths_df = pd.DataFrame({'QRS_Widths': qrs_widths})
        st.write(qrs_widths_df)
        st.write(f"Avarage QRS width: {avarage_qrs_widths} ms")
        
elif param ==("Frequency Domain Analysis"):
    st.title("Freqeuncy Domain Analysis")
    mav_window_pre=st.sidebar.number_input("MAV Window Preprocessing", value=20)
    signal_pre = movingAverage(signal, mav_window_pre)
    N = len(signal_pre)
    fs = 1000

    #Segemntasi Sinyal 
    signal_P, signal_QRS, signal_T = segmentation_p_qrs_t(signal_pre, N)
    create_plotly_figure('P-QRS-T Segmentation','Sequence (n)','Amplitude (mV)',time, [signal_P, signal_QRS, signal_T], ['Segment P', 'Segment QRS', 'Segment T'],'lines')
    
    #DFT Original Signal
    N = len(signal)
    MagDFT_full = discrete_forrier_transform(N, fs, signal)
    k =  np.arange (0, N, 1, dtype=int)
    n = np.arange (0, N, 1, dtype=int)
    create_plotly_figure('DFT Original Signal','Frequency (Hz)','Magnitude',k*fs/N, [MagDFT_full[k]], ['Original Signal'],'markers + lines')
    #Segemntasi Sinyal 
    signal_P, signal_QRS, signal_T = segmentation_p_qrs_t(signal_pre, N)
    create_plotly_figure('P-QRS-T Segmentation','Sequence (n)','Amplitude (mV)',time, [signal_P, signal_QRS, signal_T], ['Segment P', 'Segment QRS', 'Segment T'],'lines')
    MagDFT_P = discrete_forrier_transform(N, fs, signal_P)
    MagDFT_QRS = discrete_forrier_transform(N, fs, signal_QRS)
    MagDFT_T = discrete_forrier_transform(N, fs, signal_T)
    create_plotly_figure('DFT Segmentation Signal','Frequency (Hz)','Magnitude',k*fs/N, [MagDFT_P[k], MagDFT_QRS[k], MagDFT_T[k]], ['DFT Segment P', 'DFT Segment QRS', ' DFT Segment T'],'lines')
