MRI (Magnetic Resonance Imaging) and CT (Computed Tomography) are common
medical imaging techniques where its usage has been increasing since early
2000s [1]. MRI has become a primary tool for diagnosis and treatment for oncology
patients due to MR images having amazing soft-tissue contrast [2, 3]. MRI
is often favored as no ionizing radiation making it safer for repeated use however
its time consuming, expensive and less effective in imaging structures with bone
density [4–6]. CT scans followed which became ideal for radiotherapy for cancer
patients as it allowed them to find dosage delivery and organ geometry before
treatment [7], CT scans are faster and more effective at imaging bone structures
but exposes patients to ionizing radiation [8, 9]. Often images for MRI and registered
to CT to obtain benefits of both imaging techniques [10] however this
increases patient diagnosis to treatment time and increases hospital bills. Additionally,
vulnerable patients get exposed to extra ionization and radiation can
cause more harm. Therefore development in MR to synthetic CT (sCT) generation
can reduce treatment costs, reduce time for treatment and less radiation
exposure to vulnerable patients.
A publicly available dataset from SynthRAD2023 [11, 12] provides MRI and CT
images for different patients. The challenge report highlights different methodologies
of different participants. But they only explore generation of synthetic
2D images not 3D. There are many advantages of 3D images such as improved
anatomical representation for localization, improved treatment planning, accurate
volumetric assessments of tumors and reduced need for multiple scans [13–
16].
From the challenge I found a Generative Adversial Network (GAN) known as
Pix2Pix [17]. This study tests the performance of Pix2Pix and LinkNet in generating
sCT and how well it creates a 3D structure. Pix2Pix uses U-Net as the
backbone of image generation but also utilizes other steps to enhance the image.
I compare it to a LinkNet model which is designed to enhance efficiency and
residual connections helps reduce information loss also ideal for reduce computational
requirements.


References:

1. J. M. Ferguson et al., “Effect of vitamin d supplementation on all-cause mortality: A
systematic review and meta-analysis of randomized clinical trials,” JAMA, vol. 322,
no. 8, pp. 750–760, 2019.
2. M. Kachelrieß and T. Flohr, “Ct imaging,” Insights into Imaging, vol. 4, no. 2,
pp. 163–177, 2013.
3. D. Lother, M. Robert, E. Elwood, S. Smith, N. Tunariu, S. R. D. Johnston, M. Parton,
B. Bhaludin, T. Millard, K. Downey, and B. Sharma, “Imaging in metastatic
breast cancer, ct, pet/ct, mri, wb-dwi, cca: review and new perspectives,” Cancer
Imaging, vol. 23, no. 1, p. 53, 2023.
4. H. E. Team, “Mri vs. x-ray: Pros, cons, costs & more,” Healthline, 2021.
5. M. Haydon, “Ultrasound, mri and ct scan – what’s the difference?,” 2024.
6. M. C. Florkow, K. Willemsen, V. V. Mascarenhas, E. H. G. Oei, M. van Stralen,
and P. R. Seevinck, “Magnetic resonance imaging versus computed tomography for
three-dimensional bone imaging of musculoskeletal pathologies: A review,” Journal
of Magnetic Resonance Imaging, vol. 56, no. 1, pp. 11–34, 2022.
7. A. T. Davis, S. Muscat, A. L. Palmer, D. Buckle, J. Earley, M. G. J. Williams,
and A. Nisbet, “Radiation dosimetry changes in radiotherapy treatment plans for
adult patients arising from the selection of the ct image reconstruction kernel,”
BJR Open, vol. 1, no. 1, p. 20190023, 2019.
8. United States Environmental Protection Agency, “Frequent questions: Radiation
in medicine,” 2023.
9. Mayo Clinic Staff, “Ct scan,” 2023.
10. K. M. M. Touabti, F. Kharfi, K. Benkahila, and S.-A. Merouane, “Computed
tomography/magnetic resonance imaging (ct/mri) image registration and fusion
assessment for accurate glioblastoma radiotherapy treatment planning,” International
Journal of Cancer Management, vol. 13, no. 9, p. e103160, 2020.
11. A. Thummerer, E. van der Bijl, A. G. Jr, J. J. C. Verhoeff, J. A. Langendijk,
S. Both, C. A. T. van den Berg, and M. Maspero, “Synthrad2023 grand challenge
dataset: Generating synthetic ct for radiotherapy,” Medical Physics, vol. 50, no. 7,
pp. 4664–4674, 2023.
12. M. Maspero, A. Thummerer, E. van der Bijl, A. G. Jr, J. J. C. Verhoeff, J. A.
Langendijk, S. Both, and C. A. T. van den Berg, “Deep learning-based synthetic ct
generation for mri-only radiotherapy: The synthrad2023 challenge,” Medical Image
Analysis, vol. 84, p. 103276, 2024.
13. K. M. Andersson, J. Johansson, E. Norrman, and L. E. Olsson, “Clinical evaluation
of deep learning-based mr-only synthetic ct for radiotherapy planning,” Radiotherapy
and Oncology, vol. 129, no. 3, pp. 467–472, 2018.
14. B. Edwards, D. McQuaid, S. Webb, and E. Johnstone, “Mr-only radiotherapy using
synthetic ct: A multi-institutional study,” International Journal of Radiation
Oncology, Biology, Physics, vol. 109, no. 5, pp. 1452–1460, 2021.
15. R. Zhang, Z. Wang, Y. Hu, and X. Chen, “Ai-based synthetic ct generation for
mr-only radiotherapy,” Medical Physics, vol. 50, no. 5, pp. 1–15, 2023.
16. X. Han, L. S. Hibbard, A. M. O’Connell, and Y. Yu, “Deep learning for mr-to-ct
synthesis in head and neck radiotherapy,” Medical Physics, vol. 44, no. 12, pp. 5868–
5878, 2017.
17. A. Alain-Beaudoin, L. Savard, and S. Bériault, “Paired mr-to-sct translation using
conditional gans – an application to mr-guided radiotherapy,” in SynthRad2023
Challenge, 2023.
