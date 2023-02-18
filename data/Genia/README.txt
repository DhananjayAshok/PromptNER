These files are intended for the shared task at the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications:

   Genia4ERtask1.iob2:	the training data 
   sampletest1.raw:	the sample test data
   sampletest1.iob2	the reference data for the sample test data

   (the following 3 files are identical with the above 3 files except they have abstract boundary information)
   Genia4ERtask2.iob2:	the training data 
   sampletest2.raw:	the sample test data
   sampletest2.iob2:	the reference data for the sample test data

   evalIOB2.pl:	the evaluation script for producing recall, precision and F-scores

The Genia4ERtask.iob2 contains 2000 MEDLINE abstracts from the GENIA bio-named entity corpus version 3.02 which is from a controlled search using the terms {human, transcription factor, blood cell}. We have pre-processed the GENIA corpus to keep just 5 entity classes:  DNA, RNA, protein, cell_line and cell_type which are all annotated in IOB2 format.
The sampletest.raw contains 10 MEDLINE abstracts that match the last 10 abstracts in the Genia4ERtask.iob2 but the tags are removed. It is intended to show the format of test file to which the participants are supposed to apply their system.
The sampletest.iob2 is the reference file which has the correct tags.
Note that the 3 files come in 2 slightly different formats: one set with abstract boundary information and the other without. Participants can choose the convenient one.
The evalIOB2.pl will work with either format. One typical use of the script is as follows:

   evalIOB2.pl sampletest1.iob2 sampletest2.ans


SHARED TASK ORGANIZATION

Nigel Collier, National Institute of Informatics, Japan
Jin-Dong Kim, University of Tokyo, Japan
Yuka Tateisi, University of Tokyo, Japan
Tomoko Ohta, University of Tokyo, Japan
Yoshimasa Tsuruoka, University of Tokyo, Japan

CONTACT

Shared task contact: bio04sharedtask@nii.ac.jp
General workshop organization: jnlpba-request@sim.hcuge.ch
Main page: http://www.genisis.ch/~natlang/JNLPBA04/

ACKNOWLEDGEMENTS

The GENIA corpus is a product of the GENIA project which is partially supported by Information Mobility Project (CREST, JST) and Genome Information Science Project (MEXT).
