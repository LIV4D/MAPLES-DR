*****************************
MAPLES-DR Dataset Description
*****************************


The :abbr:`MAPLES-DR (MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)` project was originally motivated, back in 2018, by the absence of large fundus public datasets providing pixel-wise annotations of both anatomical and pathological structures of the retina. Such datasets would especially be valuable in a context where learning models for the automatic diagnosis  of :abbr:`DR (Diabetic Retinopathy)` and :abbr:`ME (Macular Edema)` lack explainability and interpretability.

We designed MAPLES-DR to address this gap by providing pixel-wise annotations of anatomical structures (optic disc, macula, retinal vessels) and pathological lesions (microaneurysms, hemorrhages, neovascularizations, exudates, cotton wool spots, drusens) for almost 200 images of the well known `MESSIDOR <https://www.adcis.net/en/third-party/messidor/>`_ public dataset. The annotation were conducted by a team of seven senior retinologists from Hospitals of Toronto and Montréal (Canada).

MAPLES-DR also includes grades for :abbr:`DR (Diabetic Retinopathy)` and :abbr:`ME (Macular Edema)`. These global pathological labels differs from MESSIDOR original ones as the follow the Canadian teleopthalmology screening guidelines, more similar to international standard.

.. figure:: ../_static/MAPLES-DR_Content_Overview.svg
   :width: 800px
   :align: center

   Overview of MAPLES-DR content.

:abbr:`DR (Diabetic Retinopathy)` and :abbr:`ME (Macular Edema)` grades
=======================================================================
MAPLES-DR grades for :abbr:`DR (Diabetic Retinopathy)` and :abbr:`ME (Macular Edema)` follow the guidelines developed for Canadian teleopthalmology screening. These guidelines distinguish six grades for :abbr:`DR (Diabetic Retinopathy)`:

 - **R0**: absent
 - **R1**: mild
 - **R2**: moderate
 - **R3**: severe
 - **R4A**: proliferative
 - **R4S**: stable treated proliferative
 
and three for :abbr:`ME (Macular Edema)`: 

 - **M0**: absent
 - **M1**: mild
 - **M2**: moderate 
 

Grades are defined systematically by the number and position of visible red and bright retinal lesions. Each grade is associated with a recommended course of action (from rescreening in two years for mild cases, to immediate referral to an ophthalmologist for the more severe ones). 

A detailed definition of those grades can be found in `this paper <http://doi.org/10.1016/j.jcjo.2020.01.001>`_ :cite:`Boucher2020`.

Pixel-wise annotations
======================

Anatomical structures
*********************
Anatomical structures are present in all images, including healthy ones, but their appearance and their proximity to lesions provide valuable diagnostic information.

**Retinal vessels** are indicative of the stage of :abbr:`DR (Diabetic Retinopathy)`: an increase in arteriolar tortuosity is associated with mild and moderate stages :cite:`sasongkoRetinalVascularTortuosity2011`, while venous beading and dilation are symptoms of severe proliferative stages. The vascular tree is also used as a reference to assess the readability of an image.

The **optic disc**, **optic cup**, and **macula** are also included in MAPLES-DR. Their purpose for diagnosis is two-fold. First, :abbr:`ME (Macular Edema)` is graded by counting the number of lesions within one or two optic disk diameters from the macula, which implies the annotation of both these anatomical structures. Similarly, clinical definitions of :abbr:`DR (Diabetic Retinopathy)` severity often distinguish four quadrants by diving the retina  horizontally by a line through the fovea and optic disc (superior / inferior division) and vertically by a line through the fovea (temporal / nasal division) :cite:`purvesRetinotopicRepresentationVisual2001`. Second, the positions of the lesions in relation to these healthy structures may indicate different etiologies and severities. For example, clinical guidelines sometimes distinguish between disc neovascularization and other neovascularization.


Red lesions
***********
Diabete mellitelus affects the walls of the vessels, eventually causing microvascular dysfunctions that manifest in the retina as microaneurysms, hemorrhages, intraretinal microvascular abnormalities (:abbr:`IRMA (Intra-Retinal Microvascular Abnormalities)`), or neovessels. We refer to these pathological structures as "red lesions". 

**Microaneurysms** appear as small circular dilations of the capillaries. They are early signs of microvascular dysfunction and are commonly used to detect mild :abbr:`DR (Diabetic Retinopathy)`.

Intraretinal **hemorrhages** develop in more advanced stages of the pathology and are divided into dot or blot hemorrhages. Dot hemorrhages appear as circular and well-defined spots and are typically caused by the rupture of a microaneurysm. Distinguishing them from microaneurysms is challenging, and only fundus angiography (FA) can differentiate the two with complete certainty. Blot hemorrhages are larger and have less defined borders. Both were annotated simply as *hemorrhages* in MAPLES-DR. Clinical practice also recognizes superficial (flame-shaped) and vitreous hemorrhages that appear in the most severe stages of retinopathy, none was discovered in the MAPLES-DR dataset.

Starting from the moderate non-proliferative stage (R2), irregular intraretinal vessels can appear, referred to as :abbr:`IRMA (Intra-Retinal Microvascular Abnormalities)`. The next stage of the disease (R3) coincides with even more extensive intraretinal changes, which are precursors to worsening of the disease. Indeed, the presence of :abbr:`IRMA (Intra-Retinal Microvascular Abnormalities)` indicates a 50% risk of developing **neovascularisation**  within one year, corresponding to a transition to the proliferative stage of the disease. Leakages from extensive neovascularisation are responsible for preretinal and vitreous hemorrhages that can cause major visual loss. In the fundus image, neovascularisations are difficult to distinguish from :abbr:`IRMA (Intra-Retinal Microvascular Abnormalities)`; however, fluorescein angiography may reveal a leakage that serves as a discriminant factor between the two. In the absence of this imaging modality, :abbr:`IRMA (Intra-Retinal Microvascular Abnormalities)` are not differentiated from neovascularisation in MAPLES-DR.

Bright Lesions
**************
In the severe stages of :abbr:`DR (Diabetic Retinopathy)`, the retina thickens (edema formation) and hard **exudates** (also known as lipoprotein exudation)  may appear, potentially causing loss of visual acuity. These deposits usually arise from leakage from damaged capillaries. Furthermore, in the case of ischemia, one can observe a blockage in axonal transport (the movement of mitochondria, lipids, proteins, and other substances within the neuron's body, allowing for its renewal) in the optic nerve fiber layer. 

This can lead to the appearance of lesions known as **Cotton Wool Spots** (:abbr:`CWS (Cotton Wool Spots)`), resulting from axoplasmic accumulations. They are characterized by their white appearance and blurry borders. While the principal etiology is diabetic retinopathy, :abbr:`CWS (Cotton Wool Spots)` can also be observed in other vascular diseases (systemic arterial hypertension, vein obstruction, coagulopathies...) 

Finally, MAPLES-DR also provides annotations of **drusens**. These lesions are more commonly associated with Age-related Macular Degeneration (:abbr:`AMD (Age-related Macular Degeneration)`), with a prevalence varying from 10\% (fifth decade of life) to 35\% (seventh decade).  They usually appear around the macula and are histologically situated at the interface with the Retinal Pigment Epithelium (RPE). It is supposed that they originate from degenerative products of the RPE's cells and are composed of lipids and glycoproteins. Classifying early stage :abbr:`AMD (Age-related Macular Degeneration)` depends on  estimating the size of the drusen.


Annotation Platform
===================
The annotation procedure was co-designed with the team of retinologists to meet a triple objective:

 1. Providing an intuitive yet effective annotation tool for the classification and segmentation of biomarkers in fundus images. 
 2. Enabling a collaborative effort on common annotations despite the geographical distance between the retinologists and the limited time each could dedicate to this program. 
 3. Designing a "scalable" annotation protocol, capable of being extended to much more ambitious annotation campaigns, such as labeling large Canadian telemedicine databases containing tens of thousands of images.

To meet these challenges, we developed a custom web-based annotation platform allowing the following workflow: expert annotators can access the Web portal at any time to consult and edit annotations with specialized drawing tools; these annotations and the related information (annotation times, comments) are centralized and stored in a secure database hosted on our laboratory server; as the research team, we assign tasks to annotators, monitor progress, and export annotations via a Python API. The annotation platform (portal, annotation tools, server backend, and Python API) as well as training material for annotators is available on `github <https://github.com/LIV4D/AnnotationPlatform>`_.

References
==========
.. bibliography::
   :filter: docname in docnames