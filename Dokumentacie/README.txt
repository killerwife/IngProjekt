Zdrojov� k�dy sa skladaj� z troch �ast� -
TBB kni�nica - paraleliz�cia na CPU - https://github.com/01org/tbb
OpenCV kni�nica - https://github.com/opencv/opencv
Vlastne naprogramovan� zdrojov� k�dy - https://github.com/killerwife/IngProjekt/ tu v�dy najnov�ia verzia

Pripraven� bal�k je vytvoren� pre Visual Studio 2015 a kompilovan� MSVC 14 kompil�torom. V pr�pade potreby je mo�n� upgradeova� na Visual Studio 2017 ale je vtedy potrebn� v�etky zdrojov� k�dy skompilova� s�m.
Pribalen� kni�nice s� skompilovan� s podporou TBB a taktie� podporou CUDA. Defaultn� kni�nice stiahnute�n� na str�nke s� bez oboch t�chto z�vislost�. Na kompil�ciu OpenCV a TBB sa pou��va CMAKE syst�m.
V pr�pade vlastnej kompil�cie, odpor��am stiahnu� u� hotov� stable release TBB kni�nice na githube a s�m si skompilova� OpenCV s podporou CUDA/TBB alebo pr�padne aj OpenCL. BUILD_TBB premenn� treba spr�vne nastavi�.

V pr�pade pr�ce s Visual Studiom 2015 je potrebn� nastavi� nasledovn� Enviroment Variables: (obr�zok EnviromentVariables.bmp)
OPENCV_ROOT - cesta k header s�borom OpenCV kni�nice
OPENCV_LIBRARYDIR -- cesta ku kni�niciam (.lib) OpenCV kni�nice
TBB_ROOT - cesta k header s�borom TBB kni�nice
TBB_LIBRARYDIR - cesta ku kni�niciam (.lib) TBB kni�nice

N�sledne je potrebn� urobi� jedno z dvoch mo�nost� pre TBB aj OpenCV:
a) skop�rova� .dll kni�nice do prie�inka s .exe s�borom (aj v pr�pade debugu)
b) prida� do cesty PATH cestu ku .dll kni�niciam (obr�zok PathVariable.bmp)

Po nastaven� t�chto ciest, je potrebn� zre�tartova� po��ta�, preto�e niekedy sa zmeny v ceste PATH nezmenia ihne�. V pr�pade mo�nosti a) sta�� iba vypn��/zapn�� Visual Studio.
Po otvoren� .sln v bal��ku a spr�vnom nastaven� spom�nan�ch ciest, by mal by� bal��ek plne kompilovate�n� v debug/release v x64 m�de. Na Win32 je potrebn� skompilova� v�etky zdrojov� k�dy vo Win32 kompil�ci� pre debug/release.
