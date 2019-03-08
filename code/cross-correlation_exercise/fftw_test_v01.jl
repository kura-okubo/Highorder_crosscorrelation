"""
fftw test
"""

include("./SeisJul/correlate.jl")
using Plots, .Correlate, FFTW, Test, LinearAlgebra

#------------------------------------------------#
cwave = 20		# wave speed between sensors [m/s]
dist = 1000	# distance between sensors [m]
f = 0.1		# frequency of signal [Hz]
dt = 0.1	# sampling time [s]
T = 100 	# time length of input signal [s]

#input parameters
t_init = 20	# initial time of signal at sensor 1

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

#fft

@test norm(ifft(fft(u1)) - u1) < 1e-8

p1 = plot(u1)
p2 = plot(fft(u1))
p3 = plot(irfft(fft(u1), Int(2*(length(fft(u1)))-1)))

plot(p1,p2,p3, layout = (3,1), size = (800, 1200), legend=false)