using HDF5
using Plots
using FFTW
using DSP.Periodograms
using DSP.Windows

function load_h5()
    h5open("./LDC2_sangria_training_v2.h5","r") do f
        cfg = read(f["obs/config"])
        mbhbs= read(f["sky/mbhb/tdi"])
        vgbs = read(f["sky/vgb/tdi"])
        dgbs = read(f["sky/dgb/tdi"])
        igbs = read(f["sky/igb/tdi"])
        txyz = read(f["obs/tdi"])
        cfg, mbhbs, vgbs, dgbs, igbs, txyz
    end
end

function XYZ2AET(x,y,z)
    a = (-x .+  z)./sqrt(2)
    e = (x.- 2 .* y .+z)./sqrt(6)
    t = (x .+y .+z)./sqrt(3)
    return (a,e,t)
end

cfg, mbhbs, vgbs, dgbs, igbs, txyz = load_h5()
x = [txyz[i].X for i in 1:length(txyz)]
y = [txyz[i].Y for i in 1:length(txyz)]
z = [txyz[i].Z for i in 1:length(txyz)]
a,e,t = XYZ2AET(x,y,z)

# frequency arrays
f = rfftfreq(length(a), 1/cfg["dt_tdi"])
Af = rfft(a)
Ef = rfft(e)
Tf = rfft(t)

# could plot them, but it's quite slow
#=
l = @layout [a ; b; c]
p1 = plot(f[f.>0],abs.(Af[f.>0]).^2,lw=2,label="A")
p1 = plot!(yscale=:log10,xscale=:log10,xlabel="f [Hz]", ylabel = "PSD")
p2 = plot(f[f.>0],abs.(Ef[f.>0]).^2,lw=2,label="E")
p2 = plot!(yscale=:log10,xscale=:log10,xlabel="f [Hz]", ylabel = "PSD")
p3 = plot(f[f.>0],abs.(Tf[f.>0]).^2,lw=2,label="T")
p3 = plot!(yscale=:log10,xscale=:log10,xlabel="f [Hz]", ylabel = "PSD")

plot(p1,p2,p3,layout=l)
=#


# compute a periodogram and plot that instead
wA = welch_pgram(a,256*256,0,fs=1/cfg["dt_tdi"],window=hanning)
f = freq(wA)
Apsd = power(wA)
wE = welch_pgram(e,256*256,0,fs=1/cfg["dt_tdi"],window=hanning)
Epsd = power(wE)
wT = welch_pgram(t,256*256,0,fs=1/cfg["dt_tdi"],window=hanning)
Tpsd = power(wT)
l = @layout [g ; h; i]
p1 = plot(f[f.>0],Apsd[f.>0],lw=2,label="A")
p1 = plot!(yscale=:log10,xscale=:log10,xlabel="f [Hz]", ylabel = "PSD")
p2 = plot(f[f.>0],Epsd[f.>0],lw=2,label="E")
p2 = plot!(yscale=:log10,xscale=:log10,xlabel="f [Hz]", ylabel = "PSD")
p3 = plot(f[f.>0],Tpsd[f.>0],lw=2,label="T")
p3 = plot!(yscale=:log10,xscale=:log10,xlabel="f [Hz]", ylabel = "PSD")

plot(p1,p2,p3,layout=l, leg=:bottomleft)
