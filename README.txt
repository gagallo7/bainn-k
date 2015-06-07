Falcon team
Guilherme Alcarde Gallo       z5030891
Pedro Lucas Albuquerque       z5046915

This is a program that perform the well know k-nn and w-nn machine learning algorithms in order to predict a classification data and numerical regression one, represented, respectfully, by the files ionosphere.arff and autos.arff.

Some variants of the cited algorithms have been implemented, like Value Difference Measure (VDM) and Ln-norm.

USAGE:
    python main.py

COMMANDS:
    Simple Command: <problem> <nn-type> <k-value>
        For every simple command, the VDM is disabled and the Ln-norm is 2
    Detailed Command: <problem> <nn-type> <k-value> <VDM> <Ln-norm>


EXAMPLES:
    (1) c k 5: Pick classification data and use the k-nn algorithm with k = 5
    (2) n w 3: Pick regression data and use the w-nn algorithm with k = 3
    (3) c k 5 y 2: Same of (1), but with VDM enabled
    (4) n w 3 n 8: Same of (2), but with Ln-norm equal to 8.0
