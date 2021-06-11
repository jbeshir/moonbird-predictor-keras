with (import (fetchTarball https://github.com/nixos/nixpkgs/archive/86d8a4876235f9600439401efad8b957ea3a5c26.tar.gz) {});

let
  pythonEnv = python39.withPackages (ps: with ps; [
    cython
    h5py
    joblib
    numpy
    pandas
    scikit-learn
    setuptools
    tensorflow
  ]);
in mkShell {
  buildInputs = [
    pythonEnv
  ];
}