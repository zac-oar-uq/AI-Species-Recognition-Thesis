  *	gfff??@2N
Iterator::Root::BatchV2P?s??@!??:??X@)?~?:p?@1?h??X@:Preprocessing2n
7Iterator::Root::BatchV2::Shuffle::Zip[0]::ParallelMapV2?z6?>??!U??I?2??)?z6?>??1U??I?2??:Preprocessing2\
%Iterator::Root::BatchV2::Shuffle::Zipy?&1???!
t?|n6??)M??St$??1o5pR"??:Preprocessing2{
DIterator::Root::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlicea??+e??!<??????)a??+e??1<??????:Preprocessing2W
 Iterator::Root::BatchV2::Shuffle?o_???!??$??M??)??_vO??1n?h?0]??:Preprocessing2l
5Iterator::Root::BatchV2::Shuffle::Zip[1]::TensorSlice'???????!??ͺf6??)'???????1??ͺf6??:Preprocessing2E
Iterator::Rootj?t??@!H?l06?X@)Ǻ???v?1?)?cX???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.