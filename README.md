# PJA-ASI-12C-GR2

Machine Learning Operations Projekt z przedmiotu ASI

## Tematyka projektu

### Problematyka

Celem projektu jest przewidzenie czy dany obiekt (Pokemon), po nadaniu mu pewnego zestawu wartości, zostanie uznany za "Legendarnego".
Przedwsięzięcie realizowane w ramach przedmiotu ASI na Polsko-Japońskiej Akademii Technik Komputerowych.

### Dataset

- **Źródło:** <https://github.com/pycaret/pycaret/blob/master/datasets/pokemon.csv>  
- **Liczba rekordów:** 800  
- **Liczba zmiennych:** 13  
- **Plik:** pokemon.csv  
- **Rozmiar pliku:** 44 KB  
- **Predykowana zmienna:** "Legendary"  

## Autorzy

Adam Kaczkowski - s23020
Karol Tusiński - s17580
Eryk Gregorczyk - s20454
Patryk Siedlik - s22811

## Polityka branchowania i zatwierdzania zmian w repozytorium Github

**1.** Główny branch **"main"** zawiera aktualnie obowiązujące i zatwierdzone przez wszystkich autorów wersje plików projektowych.
**2.** Dodawanie poszczególnych funkcjonalności i aktualizacje w projekcie są wdrażane według następującego schematu:
    - Autor wdrażający funkcjonalność tworzy brancha o nazwie zgodnej ze standardem: "feature/[inicjał]/[funkcjonalność]". Na przykład: "feature/eg/MachineLearning".
    - Wykonanie commitów wdrażających zmiany do brancha.
    - Utworzenie żądania zmergowania brancha z **"main"**.
    - Ręczne zatwierdzenie żądania przez wszystkich członków zespołu.
    - Zmergowanie feature brancha z main.

## Przygotowanie środowiska

**Uwaga!**
Procedura wspierana dla następujących wersji oprogramowania:

- Anaconda 2024.02-1
- Conda 24.3.0

### Instalacja Anacondy

**1.** Pobierz dystrybucję Anaconda z oficjalnej strony: <https://www.anaconda.com/download/>
**2.** Zweryfikuj hashu pliku instalacyjnego.
    - Hash pliku instalacyjnego sprawdzić na stronie: <https://docs.anaconda.com/free/anaconda/hashes/>
    - Wykonać następujące polecenie powershell, wskazując plik instalacyjny Anacondy: `Get-FileHash "Anaconda3-2024.02-1-Windows-x86_64.exe" -Algorithm SHA256`. Należy podmienić nazwę pliku, jeśli jest inna.
    - Porównaj uzyskany hash z hashem uwzględnionym na oficjalnej stronie Anacondy, podanej w punkcie **1.** procedury. Jeśli się zgadza, możesz przystąpić do realizacji kolejnego punktu procedury.
**3.** Zainstaluj Anacondę, korzystając z pobranego pliku instalacyjnego.
**4.** Zaktualizuj condę za pomoą polecenia:
`conda update conda`

### Import środowiska Conda

**1.** Z katalogu **"config"** z repozytorium projektu pobierz pliki: *"import-conda-environment.ps1"* oraz *"PJA-ASI-12C-GR2.yaml"*.
**2.** Umieść pobrane pliki w tej samej lokalizacji.
**3.** Uruchom konsolę **Anaconda Powershell** i przejdź lokalizacji, w której znajdują się pobrane pliki.
**4.** Uruchomić skrypt *"import-conda-environment.ps1"*.
**5.** Po poprawnym imporcie środowsika, w konsoli powinien pojawić się komunikat: "Environment PJA-ASI-12C-GR2 created and activated.".
