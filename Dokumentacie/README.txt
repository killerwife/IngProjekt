Zdrojové kódy sa skladajú z troch èastí -
TBB kninica - paralelizácia na CPU - https://github.com/01org/tbb
OpenCV kninica - https://github.com/opencv/opencv
Vlastne naprogramované zdrojové kódy - https://github.com/killerwife/IngProjekt/ tu vdy najnovšia verzia

Pripravenı balík je vytvorenı pre Visual Studio 2015 a kompilovanı MSVC 14 kompilátorom. V prípade potreby je moné upgradeova na Visual Studio 2017 ale je vtedy potrebné všetky zdrojové kódy skompilova sám.
Pribalené kninice sú skompilované s podporou TBB a taktie podporou CUDA. Defaultné kninice stiahnute¾né na stránke sú bez oboch tıchto závislostí. Na kompiláciu OpenCV a TBB sa pouíva CMAKE systém.
V prípade vlastnej kompilácie, odporúèam stiahnu u hotovı stable release TBB kninice na githube a sám si skompilova OpenCV s podporou CUDA/TBB alebo prípadne aj OpenCL. BUILD_TBB premennú treba správne nastavi.

V prípade práce s Visual Studiom 2015 je potrebné nastavi nasledovné Enviroment Variables: (obrázok EnviromentVariables.bmp)
OPENCV_ROOT - cesta k header súborom OpenCV kninice
OPENCV_LIBRARYDIR -- cesta ku kniniciam (.lib) OpenCV kninice
TBB_ROOT - cesta k header súborom TBB kninice
TBB_LIBRARYDIR - cesta ku kniniciam (.lib) TBB kninice

Následne je potrebné urobi jedno z dvoch moností pre TBB aj OpenCV:
a) skopírova .dll kninice do prieèinka s .exe súborom (aj v prípade debugu)
b) prida do cesty PATH cestu ku .dll kniniciam (obrázok PathVariable.bmp)

Po nastavení tıchto ciest, je potrebné zreštartova poèítaè, pretoe niekedy sa zmeny v ceste PATH nezmenia ihneï. V prípade monosti a) staèí iba vypnú/zapnú Visual Studio.
Po otvorení .sln v balíèku a správnom nastavení spomínanıch ciest, by mal by balíèek plne kompilovate¾nı v debug/release v x64 móde. Na Win32 je potrebné skompilova všetky zdrojové kódy vo Win32 kompilácií pre debug/release.
