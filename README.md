# Grand Challenge: Live Video Streaming

This repository contains the simulator code being used for Live Video Streaming.  The simulator is a frame-level DASH video streaming simulator with a pluggable client module for bitrate control and latency control.

## Files

![Image text](https://github.com/tse-hou/Live-Video-Streaming-Challenge/blob/master/architecture.gif)

The simulator contains the following files:

* `run.py` : An SDK to call the SIM and ABR
* `ABR.py` : your ABR algorithm
* `fixed_env.py` : SIM code simulates live streaming player download, play, card, skip frame, etc
* `load_trace.py` : load trace to memory

The folder `dataset` contains the network traces and video traces.  Please see the README.md file in the dataset directory for decsription of the dataset.

# The Simulator

The simulator simulates a live video player，including downloading and playback of video frames.  It reads as inputs:

1. A video trace, which contains the size of each video frame in a video file.
2. A network trace, which contains the network throughput at each time instance.
3. The bitrate control and latency control algorithm provided by participant.

The simulator output the following:

|   params           | params description                       |  example   |
| ------------------ | ---------------------------------------- | ---------- |
| time(s)            | physical time                            |   0.46(s)  |
| time_interval(s)   | duration in this cycle                   |   0.012(s) |  
| send_data_size(bit)| The data size downloaded in this cycle   |   14871(b) |
| frame_time_len(s)  | The time length of the frame currently   |   0.04(s)  |
| rebuf(s)           | The rebuf time of this cycle             |   0.00(s)  |
| buffer_size(s)     | The buffer size time length              |   1.26(s)  |
| play_time_len(s)   | The time length of playing in this cycle |   0.012(s) |
| end_delay(s)       | Current end-to-to delay                  |   1.31(s)  |
| cdn_newest_id      | Cdn the newest frame id                  |   85       |
| download_id        | Download frame id                        |   41       |
| cdn_has_frame      | cdn cumulative frame info                |   1.31(s)  |
| decision_flag      | Gop boundary flag or I frame flag        |   False    |
| buffer_flag        | Whether the player is buffering          |   False    |
| cdn_rebuf_flag     | Whether the cdn is rebuf                 |   False    |
| end_of_video       | Whether the end of video                 |   False    |

## Requirement

You will need Python 3 to run the simulator.

## Running the Simulator

To run the simulator, you execute

```
python run.py all
```

The given default code should produce something like the following:

```
AsianCup_China_Uzbekistan,fixed: Start
AsianCup_China_Uzbekistan,low: Start
AsianCup_China_Uzbekistan,high: Start
AsianCup_China_Uzbekistan,medium: Start
Fengtimo_2018_11_3,fixed: Start
Fengtimo_2018_11_3,low: Start
Fengtimo_2018_11_3,medium: Start
Fengtimo_2018_11_3,high: Start
game,fixed: Start
game,low: Start
game,medium: Start
game,high: Start
room,fixed: Start
room,low: Start
room,medium: Start
room,high: Start
sports,fixed: Start
sports,medium: Start
sports,low: Start
sports,high: Start
YYF_2018_08_12,fixed: Start
YYF_2018_08_12,medium: Start
YYF_2018_08_12,low: Start
YYF_2018_08_12,high: Start
game,low: Done
YYF_2018_08_12,medium: Done
room,fixed: Done
room,medium: Done
game,medium: Done
AsianCup_China_Uzbekistan,fixed: Done
game,fixed: Done
Fengtimo_2018_11_3,fixed: Done
YYF_2018_08_12,low: Done
Fengtimo_2018_11_3,medium: Done
Fengtimo_2018_11_3,low: Done
room,low: Done
sports,medium: Done
sports,fixed: Done
sports,low: Done
AsianCup_China_Uzbekistan,high: Done
Fengtimo_2018_11_3,high: Done
AsianCup_China_Uzbekistan,medium: Done
YYF_2018_08_12,fixed: Done
room,high: Done
AsianCup_China_Uzbekistan,low: Done
YYF_2018_08_12,high: Done
game,high: Done
sports,high: Done
[[1003.0643755032955, 1.9479399923825436e-06], [1468.1899223244538, 2.6868329406780786e-06], [1515.4896973625787, 2.2991288007135838e-06], [1535.3410570279666, 2.679303495996842e-06], [996.4359586392395, 2.47624286579198e-06], [1360.1296820536336, 2.5919058099186546e-06], [1634.203817372895, 2.192868066774833e-06], [1944.3635834321335, 2.025828069570113e-06], [962.1517246733974, 2.6871058158121116e-06], [1391.4320800027003, 2.459983321487817e-06], [1632.4709275956995, 2.6005579382470035e-06], [1862.3119642996699, 2.2600745894218404e-06], [1022.4077588924314, 2.7316122010381435e-06], [1399.8878565939472, 2.6031690445944086e-06], [1647.239833065784, 2.6443942648465522e-06], [1867.1238357565812, 2.3751636671889677e-06], [918.3925549314956, 2.2861276057325095e-06], [1268.203516951356, 2.0833536612386818e-06], [1303.9777877439697, 2.7589547541058594e-06], [1324.7899687878103, 2.7192239646288926e-06], [963.301917210777, 2.6875973646716527e-06], [1415.9670930158863, 2.6465205031324433e-06], [1619.5561920445957, 2.696355642315498e-06], [1846.2505164302045, 2.6341100208094863e-06]]
score:  [1.41261182e+03 2.49059810e-06]

```

## Configuring the Simulator

* run all video_trace and network_trace
```
python run.py all
```

* run specific video_trace and network_trace
```
python run.py [video_trace] [network_trace]
```

* The simulator can produce detailed log files for debugging.  To turn this on, set the variable `DEBUG` to `True`.  By default, the logs will be written to a sub-directory called `log`.  This log directory can be changed by setting the `LOG_FILE_PATH` variable.  Note that you may want to set `DEBUG` to `False` if you are training an AI model as large volume of data may be written to disk when logging is on.

## Reference
- Website: https://www.aitrans.online/MMGC/
- Paper of simulator: https://dl.acm.org/doi/abs/10.1145/3343031.3356083
- Paper of others ABR for Live video streaming: https://dl.acm.org/doi/proceedings/10.1145/3343031#heading39