	?P??"Q@?P??"Q@!?P??"Q@	??K?????K???!??K???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?P??"Q@2Ƈ????A?&S?P@Y??*?)??*	gffff??@2F
Iterator::Model?c?? ???!??'??V@)?J??^b??1wS??xWV@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap,}??????!Z?/q?@)3??p?ܜ?1>,N??q
@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?1Xq????!?????@)?ڦx\T??1k? 2N
	@:Preprocessing2U
Iterator::Model::ParallelMapV26#??E???!??t\?	@)6#??E???1??t\?	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice? ?S?Dy?!??F??&??)? ?S?Dy?1??F??&??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorW$&??[x?!^J$?Q??)W$&??[x?1^J$?Q??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Ȳ`⏲?!?/`??!@)'i??v?16R5?h???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??K???I??o?ҭX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2Ƈ????2Ƈ????!2Ƈ????      ??!       "      ??!       *      ??!       2	?&S?P@?&S?P@!?&S?P@:      ??!       B      ??!       J	??*?)????*?)??!??*?)??R      ??!       Z	??*?)????*?)??!??*?)??b      ??!       JCPU_ONLYY??K???b q??o?ҭX@