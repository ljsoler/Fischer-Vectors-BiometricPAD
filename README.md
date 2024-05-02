#### This is the official repository of the papers:
1. ###  Fingerprint Presentation Attack Detection Based on Local Features Encoding for Unknown Attacks [IEEE Access](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9312046) 
2. ### Local Feature Encoding for Unknown Presentation Attack Detection: An Analysis of Different Local Feature Descriptors [IET Biometrics](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/bme2.12023)
3. ### On the Generalisation capabilities of Fisher Vector based Face Presentation Attack Detection [IET Biometrics](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/bme2.12041)
4. ### Fisher Vectors for Biometric Presentation Attack Detection [Hand of Biometric Antispoofing](https://drive.google.com/file/d/1TkWUZ5vDmsQ8WMhdWM7ffMXESN16Vh8R/view)


## Requirements
- opencv > 2.410
- vlfeat > 0.9

## Solution description

### Fingerprint

* repo_name: SIFTEncodings

Info: The algorithm encodes local SIFT features using one of three representations (i.e., FV, BoW, and VLAD).

    - usage: ./PDA_sdk -h for help


* repo_name: TextureDescFV

Info: The algorithm encodes different texture descriptor (e.g., continuous and binary descriptors) using Fisher Vector representation.

    - usage: ./PDAfv -h for help


### Face

* repo_name: FVencBSIF

Info: The algorithm encodes compact BSIF features, extracted from local patches, with the FV representation.

    - usage: ./PDAlocalEnc -h for help

<hr/>

## Citation ##
If you use any of the code provided in this repository or the models provided, please cite the following paper:
```
@article{GonzalezSoler-FingerprintPADonLocalFeatures-IEEE-Access-2021,
 Author = {L. J. Gonzalez-Soler and M. Gomez-Barrero and L. Chang and A. Perez-Suarez and C. Busch},
 Groups = {ATHENE, RESPECT, CRISP},
 Journal = {{IEEE} Access},
 Keywords = {Fingerprint Recognition, Presentation Attack Detection},
 Month = {January},
 Pages = {5806--5820},
 Title = {Fingerprint Presentation Attack Detection Based on Local Features Encoding for Unknown Attacks},
 Volume = {9},
 Year = {2021}
}

@article{GonzalezSoler-FVUnkownAttacksLivDet-IET-Biometrics-2021,
 Author = {L. J. Gonzalez-Soler and M. Gomez-Barrero and J. Kolberg and L. Chang and A. Perez-Suarez and C. Busch},
 Groups = {ATHENE, RESPECT, CRISP, NGBS},
 Journal = {{IET} Biometrics},
 Keywords = {Fingerprint Recognition, Presentation Attack Detection},
 Number = {4},
 Pages = {374--391},
 Title = {Local Feature Encoding for Unknown Presentation Attack Detection: An Analysis of Different Local Feature Descriptors},
 Volume = {10},
 Year = {2021}
}

@article{GonzalezSoler-UnkownAttacksFace-IET-Biometrics-2021,
 Author = {L. J. Gonzalez-Soler and M. Gomez-Barrero and C. Busch},
 Groups = {ATHENE, RESPECT, CRISP, NGBS},
 Journal = {{IET} Biometrics},
 Keywords = {Face Recognition, Presentation Attack Detection},
 Month = {September},
 Number = {5},
 Pages = {480--496},
 Title = {On the Generalisation capabilities of Fisher Vector based Face Presentation Attack Detection},
 Volume = {10},
 Year = {2021}
}

@incollection{GonzalezSoler-PAD-FVBioPAD-Springer-2023,
 Author = {Gonzalez-Soler, Lazaro Janier and Gomez-Barrero, Marta and Patino, Jose and Kamble, Madhu and Todisco, Massimiliano and Busch, Christoph},
 Booktitle = {Handbook of Biometric Anti-Spoofing: Presentation Attack Detection and Vulnerability Assessment},
 Groups = {ATHENE, RESPECT},
 Keywords = {Presentation Attack Detection},
 Pages = {489--519},
 Publisher = {Springer},
 Title = {Fisher Vectors for Biometric Presentation Attack Detection},
 Year = {2023}
}
```

