"""
High order cross correlation (C2, C3) exercise

March 07, 2019
Kurama Okubo
"""

using Plots, FFTW, LinearAlgebra

#------------------------------------------------#
lx = 1e4	# distance of sensors R1 = (-lx, 0) R2 = (+lc, 0) [m]
R = 2e4	# radius of source location (S = (Rcos(theta), Rsin(theta))) [Hz]
vel= 3e3 # wave speed between sensors [m/s]
rho = 3000.0	# density of medium [kg/m3]
Q = 1e9 	# Attenuation term (quality factor) Q
dt = 0.01
T = 50
NumofSource = 50;

ifRicker = true# If Ricker wavelet is convolved
ifcheckRicker = false #debug Ricker wavelet
fM = 5 # [Hz] peak (center) frequency
#------------------------------------------------#


#The far field Green function (Fichtner, 2015 (doi:10.1093/gji/ggv182) eq. 12)

"""
gf(x::Array{Float64,1},xi::Array{Float64,1},omega::Float64, vel::Float64, rho::Float64, Q::Float64)
returns the Green’s function solution for two-dimensional space, under the far field approximation
"""
function gf(x::Array{Float64,1},xi::Array{Float64,1},omega::Float64, vel::Float64, rho::Float64, Q::Float64)
    r = norm(x.-xi)
    g1 = (1/sqrt((8*pi*omega*(1/vel) * r))) * exp(-im * (omega* (1/vel) * r + pi/4)) * exp(-(omega* r)/(2*vel*Q)) 
    return g1
end


"""
rickerWavelet(omega)
returns zero-phase Ricker wavelet
"""
function rickerWavelet(omega::Float64, omega0::Float64)
    ricker = ((2*omega^2)/(sqrt(pi) * omega0^3)) * exp(-(omega^2 / omega0^2))
    return ricker
end


#location of receiver 1 and 2
Rx1 = [-lx, 0]
Rx2 = [lx, 0]

#location of receiver 3
Rx3 = [-2*lx, 0]

#location of source
theta = zeros(NumofSource, 1)
Sx = zeros(NumofSource, 2)
for i = 1:NumofSource
    theta[i] = 2*pi*(i-1)/NumofSource
    Sx[i,:] = R .* [cos(theta[i]), sin(theta[i])]  
end

#omega
NT = round(Int64, T/dt)
NumofFreq = NT
fs = 1/dt
omega = 2*pi.*(fs*(0:(NT/2))/NT);

gf1 = zeros(Complex{Float64}, NumofSource, 2*(length(omega)-1))
gf2 = zeros(Complex{Float64}, NumofSource, 2*(length(omega)-1))
gf3 = zeros(Complex{Float64}, NumofSource, 2*(length(omega)-1))

for i = 1:NumofSource
    for j = 1:length(omega)-1
        gf1[i,j] = gf(Rx1, Sx[i,:], omega[j], vel, rho, Q)
        gf2[i,j] = gf(Rx2, Sx[i,:], omega[j], vel, rho, Q)
        gf3[i,j] = gf(Rx3, Sx[i,:], omega[j], vel, rho, Q)
        
        #remove components at omega = zero

        gf1[i,1] = 0.0
        gf2[i,1] = 0.0
        gf3[i,1] = 0.0

    end
end

if ifRicker
    for i = 1:NumofSource
        for j = 1:length(omega)-1
            gf1[i,j] = gf1[i,j] * rickerWavelet(omega[j], 2*pi*fM)
            gf2[i,j] = gf2[i,j] * rickerWavelet(omega[j], 2*pi*fM)
            gf3[i,j] = gf3[i,j] * rickerWavelet(omega[j], 2*pi*fM)

        end
    end
end


#make it symmetric
for i = 1:1
    for j = 1:length(omega)-1
        gf1[i, Int((length(omega)-1) + j)] = real(gf1[i, Int(length(omega) - j)]) + im*-imag(gf1[i, Int(length(omega) - j)]);
        gf2[i, Int((length(omega)-1) + j)] = real(gf2[i, Int(length(omega) - j)]) + im*-imag(gf2[i, Int(length(omega) - j)]);
        gf3[i, Int((length(omega)-1) + j)] = real(gf3[i, Int(length(omega) - j)]) + im*-imag(gf3[i, Int(length(omega) - j)]);
    end
end


if ifcheckRicker
    """
    check Ricker wavelet
    """
    rwave = zeros(Complex{Float64}, 2*(length(omega)-1))
    for j = 1:length(omega)
        rwave[j] = rickerWavelet(omega[j], 2*pi*fM);
    end
    fs_ricker = omega ./ (2*pi)
    rp1 = plot(fs_ricker, real(rwave)[1:length(omega)], line=(:black, 1, :solid),
        xlabel = "Frequency[Hz]", 
        ylabel = "Spectral density",
        title = "Ricker wavelet",
       )

    URicker = fftshift(ifft(rwave))
    t_ricker = dt*range(-(length(omega)-1),(length(omega)-1))
    TD = sqrt(6)/(pi*fM)
    TR = TD/sqrt(3)
    rp2 = plot(t_ricker[1:end-1], real(URicker), line=(:black, 1, :solid),
        xlabel = "Time [s]", 
        ylabel = "u1 [m]",
        xlim = (-3*TD, 3*TD),
        )
    rp2 = plot!([TD, TD]./2, [-1, 1] .* 1e-3, line=(:red, 1, :dash))
    rp2 = plot!([TR, TR]./2, [-1, 1] .* 1e-3, line=(:blue, 1, :dash))
    plot(rp1, rp2, layout = (2,1), size = (800, 800), legend=false)
end

"""
Exercise 0: Plot displacement in time domain
"""

# take inverse fft
u1test = ifft(gf1[1,:])
t = dt .* range(0, length(omega)-1)

p1 = plot(t, real.(u1test[1:length(omega)]), line=(:black, 1, :solid),
    #marker = (:cross, 2, :green),
    xlabel = "Time [s]", 
    ylabel = "u1 [m]",
    title = "u1",
    xlim = (0, 12),
    ylim = (1.2*minimum(real.(u1test[1:length(omega)])), maximum(real.(u1test[1:length(omega)])) * 1.2)
    )

plot(p1, layout = (1,1), size = (600, 600), legend=false)

# plot along azimuth
signal_magnification = 1e6
maxplotT = 12
xticks = 0:45:360

p_all = plot(xlabel = "azimuth [deg]", 
    ylabel = "Time [s]",
    title = "Displacement at R1",
    xlim = (-20, 380),
    ylim = (minimum(t), maxplotT),
    xticks = xticks
    )

for i = 1:NumofSource
    u1test = ifft(gf1[i,:])
    u2test = ifft(gf2[i,:])
    azimuth =  rad2deg(theta[i])  
    p_all = plot!(azimuth.+ signal_magnification.*real.(u1test[1:length(omega)]), t, line=(:red, 1, :solid))
    p_all = plot!(azimuth.+ signal_magnification.*real.(u2test[1:length(omega)]), t, line=(:blue, 1, :solid))
end

#plot at 360
u1last = ifft(gf1[1,:])
u2last = ifft(gf2[1,:])
azimuthlast = 360
p_all = plot!(azimuthlast.+ signal_magnification.*real.(u1last[1:length(omega)]), t, line=(:red, 1, :solid))
p_all = plot!(azimuthlast.+ signal_magnification.*real.(u2last[1:length(omega)]), t, line=(:blue, 1, :solid))


plot(p_all, layout = (1,1), size = (600, 600), legend=false)

"""
Exercise 1: First order cross correlation C1:1->2
"""

cc1_12 = zeros(Complex{Float64}, NumofSource, 2*(length(omega)-1))

for i = 1:NumofSource
    for j = 1:2*(length(omega)-1
        
        cc1_12[i,j] = conj(gf1[i,j]) * gf2[i,j]

    end
end


# plot along azimuth
signal_magnification = 1e6
maxplotT = 12
xticks = 0:45:360

p_all = plot(xlabel = "azimuth [deg]", 
    ylabel = "Time [s]",
    title = "Displacement at R1",
    xlim = (-20, 380),
    ylim = (minimum(t), maxplotT),
    xticks = xticks
    )

for i = 1:NumofSource
    u1test = ifft(gf1[i,:])
    u2test = ifft(gf2[i,:])
    azimuth =  rad2deg(theta[i])  
    p_all = plot!(azimuth.+ signal_magnification.*real.(u1test[1:length(omega)]), t, line=(:red, 1, :solid))
    p_all = plot!(azimuth.+ signal_magnification.*real.(u2test[1:length(omega)]), t, line=(:blue, 1, :solid))
end

#plot at 360
u1last = ifft(gf1[1,:])
u2last = ifft(gf2[1,:])
azimuthlast = 360
p_all = plot!(azimuthlast.+ signal_magnification.*real.(u1last[1:length(omega)]), t, line=(:red, 1, :solid))
p_all = plot!(azimuthlast.+ signal_magnification.*real.(u2last[1:length(omega)]), t, line=(:blue, 1, :solid))


plot(p_all, layout = (1,1), size = (600, 600), legend=false)



"""
Exercise 2: Second order cross correlation C2:1->2
"""

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

