# Symulacja problemu trzech ciał (N-body, N=3) + animacja MP4
To symulacja klasycznego "tree-body problem", czyli ruchu trzech ciał oddziałujących grawitacyjnie w 2D. Symulacja zapisuje animację trajektorii do pliku MP4.
Symulacja wykorzystuje prosty schemat całkowania drugiego rzędu (wariant Verleta/ różnic centralnych) na podstawie pozycji `x,y` oraz „poprzednich” pozycji `x_prev, y_prev`.

## krótki opis skryptu:
- Ustawia warunki początkowe dla 3 ciał (pozycje i prędkości).
- W każdej iteracji liczy siły grawitacyjne pomiędzy parami ciał:  
  \( F \propto \frac{1}{r^2} \) (stała `G` ustawiona na 1.0).
- Aktualizuje pozycje metodą:
  - `x_new = 2x - x_prev + fx*dt^2`
  - `y_new = 2y - y_prev + fy*dt^2`
- Zapisuje klatki animacji do pliku `animacja_do_trajektorii_8.mp4`:
  - tło białe
  - ślady trajektorii jako małe punkty
  - aktualne pozycje jako większe okręgi
  - licznik kroku w lewym górnym rogu

## Wymagania
- Python 3.x
- numpy
- matplotlib 
- opencv-python (do generowania wideo MP4)

## Uwaga:
To uproszczony model (bez mas, masy domyślnie traktowane jak jednakowe).
Dla bardzo małych odległości dodano + 1e-10, żeby nie było dzielenia przez zero.
