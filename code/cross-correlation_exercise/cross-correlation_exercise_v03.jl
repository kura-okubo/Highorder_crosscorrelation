"""
High order cross correlation (C2, C3) exercise

March 07, 2019
Kurama Okubo
"""

include("./SeisJul_vkurama/SeisJul.jl")
include("./SeisJul_vkurama/correlate_kurama.jl")
using Plots, .Correlate, FFTW

#------------------------------------------------#
cwave = 3000 # wave speed between sensors [m/s]
lx = 1e3	# distance of sensors R1 = (-lx, 0) R2 = (+lc, 0) [m]
R = 1e4	# radius of source location (S = (Rcos(theta), Rsin(theta))) [Hz]
rho = 3000	# density of medium [kg/m3]
Q = 0 	# Attenuation term (quality factor) Q

#------------------------------------------------#

# Making time series (pseudo inputs)

N = round(Int, T/dt + 1) # number of data point 

t = dt .* collect(0:N-1) # Time [s]

u1 = zeros(N)
u2 = zeros(N)

period = 1/f

if period < dt
	println("dt is too large.")
	return
end

init_id = round(Int, t_init/dt) + 1
delay = dist/cwave #[s]

for i = 1:round(Int, period/dt)
	u1[i+init_id] = sin(2*pi*f*dt*i)
	u2[i+init_id + round(Int,delay/dt)] = sin(2*pi*f*dt*i)
end

p1 = plot(t, u1, line=(:black, 1, :solid),
	marker = (:cross, 2, :green),
    ylabel = "Signal", 
    xlabel = "Time [s]",
    title = "Station 1",
    xlim = (0, T),
    ylim = (-1.5, 1.5)
    )

p2 = plot(t, u2, line=(:blue, 1, :solid),
	marker = (:cross, 2, :green),
    ylabel = "Signal", 
    xlabel = "Time [s]",
    title = "Station 2",
    xlim = (0, T),
    ylim = (-1.5, 1.5)
    )

#Do cross-correlation in time domain
#add zero for the length of N-1 to both sides of u2
#normalized by the number of data points N-k (k is the data number in zero padding)

Rcorr	= zeros(Float64, 2*N-1)
u2corr	= vcat(zeros(N-1), u2, zeros(N-1)) 
k 		= vcat(collect(N-1:-1:0), collect(1:N-1))

for n = 1:2*N-1

	#scale = 1/(N-k[n]) #scale by 1/(N-k)
	scale = 1/N

	for m = 1:N

		Rcorr[n] += scale * u1[m] * u2corr[n+m-1]

	end
end

#time axis is determined by the sampling rate
tcorr = dt .* collect(-(N-1):(N-1))

p3 = plot(tcorr, Rcorr, line=(:red, 1, :solid),
	marker = (:cross, 2, :green),
    ylabel = "Correlation", 
    xlabel = "Time [s]",
    title = "Cross-correlation between u1 and u2: direct",
    xlim = (-T, T),
    ylim = (-1.2*maximum(abs.(Rcorr)), 1.2*maximum(abs.(Rcorr)))
    )

#Do cross-correlation in frequency domain

Fs = 1/dt

#padding zero with next2pow
u1pad = vcat(u1, zeros(nextpow(2,2*N)-N))
u2pad = vcat(u2, zeros(nextpow(2,2*N)-N))

L = length(u1pad)

#Plot single side spectrum
Fu1 = fft(u1pad)
Pu1_temp =  abs.(Fu1/L)
Pu1 = Pu1_temp[1:Int(L/2 + 1)]
Pu1[2:end-1] = 2 .* Pu1[2:end-1]
freq_Fu1 = Fs .* collect(0:Int(L/2)) ./ L

p4 = plot(freq_Fu1, Pu1, line=(:red, 1, :solid),
	marker = (:cross, 2, :green),
    ylabel = "|P1(f)|", 
    xlabel = "Frequency [Hz]",
    title = "Single-sided amplitude spectrum of u1",
    xlim = (0, Fs/2),
    xticks = 0:0.1:Fs/2,
    ylim = (0, 1.2*maximum(abs.(Pu1)))
    )

Fu2 = fft(u2pad)
Pu2_temp =  abs.(Fu2/L)
Pu2 = Pu2_temp[1:Int(L/2 + 1)]
Pu2[2:end-1] = 2 .* Pu2[2:end-1]
freq_Fu2 = Fs .* collect(0:Int(L/2)) ./ L

p5 = plot(freq_Fu2, Pu2, line=(:red, 1, :solid),
	marker = (:cross, 2, :green),
    ylabel = "|P2(f)|", 
    xlabel = "Frequency [Hz]",
    title = "Single-sided amplitude spectrum of u2",
    xlim = (0, Fs/2),
    xticks = 0:0.1:Fs/2,
    ylim = (0, 1.2*maximum(abs.(Pu2)))
    )

# Calculate cross-correlation function
# without and with zeropad

Fu1xcorr_nopad = fft(u1)
Fu2xcorr_nopad = fft(u2)

Fu1xcorr_zeropad = fft(u1pad)
Fu2xcorr_zeropad = fft(u2pad)

Rcorr_byfft_nopad, tn_nopad = Correlate.xcorrwithDFT(Fu1xcorr_nopad, Fu2xcorr_nopad, method="ddeconv")
Rcorr_byfft_zeropad, tn_zeropad = Correlate.xcorrwithDFT(Fu1xcorr_zeropad, Fu2xcorr_zeropad, method="ddeconv")

tcorr_nopad = dt .* collect(tn_nopad)
tcorr_zeropad = dt .* collect(tn_zeropad)

p6 = plot(tcorr_nopad, real.(Rcorr_byfft_nopad), line=(:brue, 1, :solid),
	marker = (:cross, 2, :green),
    ylabel = "Correlation", 
    xlabel = "Time [s]",
    title = "Cross-correlation by FFT without zero pad",
    xlim = (-T, T),
    ylim = (-1.2*maximum(abs.(Rcorr_byfft)), 1.2*maximum(abs.(Rcorr_byfft)))
    )

p7 = plot(tcorr_zeropad, real.(Rcorr_byfft_zeropad), line=(:brue, 1, :solid),
    marker = (:cross, 2, :green),
    ylabel = "Correlation", 
    xlabel = "Time [s]",
    title = "Cross-correlation by FFT with zero pad",
    xlim = (-T, T),
    ylim = (-1.2*maximum(abs.(Rcorr_byfft)), 1.2*maximum(abs.(Rcorr_byfft)))
    )

p8 = plot(tcorr, Rcorr./(maximum(Rcorr)), line=(:black, 2, :solid),
    ylabel = "Correlation", 
    xlabel = "Time [s]",
    )

p8 = plot!(tcorr_zeropad, real.(Rcorr_byfft_zeropad)./ maximum(real.(Rcorr_byfft_zeropad)), line=false,
    marker = (:xcross, 2, :red),
    ylabel = "Correlation", 
    xlabel = "Time [s]",
    title = "Comparison between direct and using FFT",
    xlim = (-T, T),
    ylim = (-1.2,1.2)
    )


plot(p1, p2, p4, p5, p3, p6, p7, p8, layout = (4,2), size = (1200, 1200), legend=false)

savefig("./summary.png")

