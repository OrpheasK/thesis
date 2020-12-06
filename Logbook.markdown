- Download [Lakh Pianoroll Daraset](https://salu133445.github.io/lakh-pianoroll-dataset/)

lpd-17-cleansed dataset with genre data matched from id_lists_tagtraum.tar.gz

Code used: -

- Εξαγωγή ονμάτων φακέλων από το dataset για διασταύρωση IDs

Δημιουργία txt (test2.txt) με τα ονόματα φακέλων του lpd_17_cleansed για σύγκριση με τα tagtraum IDs τύπου TR...

Code used: walkos.py

- Εύρεση κομματιών ανα είδος στο lpd_17_cleansed και μεταφορά τους

Χρήση test2.txt και id_list_(genre).txt για δημιουργία txt (genre_cleansed.txt) με τα TR IDs συγκεκριμένου είδους

Μεταφορά των npz αρχείων σε φακέλους ανα είδος (genre_cleansed)

Code used: compare.py, move.py, findfiles.py

- Μετατροπή είδους αρχείων και εύρεση αναπαράστασης

Μετατροπή των npz αρχείων σε midi

Αναπαράσταση γεγονότων midi με απλουστευμένα διανύσματα (time, note_duration, note, velocity(?))

Code used: npztomidi.py, midiwav.py, midicsv.py

- Εισαγωγή του music21 toolkit

Μετατροπή των midi αρχείων σε csv βάσει του music21 πρωτοκόλλου

Αναπαράσταση κάθε midi event με διανύσματα (offset, quarterlength, pitch) με αντίστοιχες μονάδες μέτρησης (num of quarter notes, -//-, midi pitch)

Μετατροπή της csv μορφής σε midi file

Code used: midi21txt.py, txtmidi21.py

-Πρώτα πειράματα Μηχανικής Μάθησης

Χρήση του Keras Functional API

Σχεδίαση autoencoder, χρήση teacher forcing αλγορίθμου

Code used: fun.ipynb, funaut.ipynb, funtfrc.ipynb, tfrc_ptch.ipynb
