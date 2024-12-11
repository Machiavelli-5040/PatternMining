# PatternMining

This is a repository containing my code that I wrote during my M2 internship at the
Centre Borelli.

When importing the results from a Keypoint-MoSeq project, and after creating the
compressed representations of all folders, your project should look like this:

```
datasets/
    <dataset_name_0>/
        <penalty_value_0>/
            standard/
                0.csv
                1.csv
                ...
            compressed/
                0.csv
                1.csv
                ...
        <penalty_value_1>/
            standard/
                0.csv
                1.csv
                ...
            compressed/
                0.csv
                1.csv
                ...
        ...
    ...
```

The path to a folder having the name as one of your Keypoint-MoSeq projects (such as
`<dataset_name_0`> above) is called a `dataset_path`. If a path points to a folder
having a penalty value as name (such as `<penalty_value_1>` above), it is called a
`penalty_path`. Finally, if it points to a folder called `standard` or `compressed`, it
is called a `mode_path`.

The standard representation of a syllable sequence is the same as the `syllable` column
of a Keypoint_MoSeq result file, and its compressed version adds two columns `frame`
(starting frame) and `duration` (length of the syllables in frames) for easier
visualization. For example, the compressed version of the following extract:
```
syllable
0
0
0
1
1
1
1
0
0
2
2
2
```
is:
```
frame,syllable,duration
0,0,3
3,1,4
7,0,2
9,2,3
```