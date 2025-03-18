set terminal gif animate delay 10
set output 'animation.gif'
set pm3d map
set palette rgbformulae 22,13,10
set cbrange [0:1]
dx = 1.0 / 999
dy = 1.0 / 999
do for [i=0:100] {
    splot sprintf('frame_%03d.txt', i) matrix using ($1*dx):($2*dy):3 with pm3d
}
