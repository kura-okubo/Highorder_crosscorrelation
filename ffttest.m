clf

startID = 41;
y = zeros(100,1);
y(startID:startID+2) = sin(2*pi*([1:3]/3));

subplot(3,1,1)
plot(y);

Y = fft(y);

subplot(3,1,2)

%Y(51:100) = 0

plot(real(Y))
hold on;
plot(imag(Y))


x = ifft(Y);

subplot(3,1,3)
plot(real(x))