	?R??F?T@?R??F?T@!?R??F?T@	'ЮI2C??'ЮI2C??!'ЮI2C??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?R??F?T@N??oD???A~?D??T@Y?~?nض??*	0?$?x@2F
Iterator::Modelٲ|]????!?)?CQT@)?Q???1????fR@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatsd?????!?8䠠)@)U0*?Ф?1&@?ic%%@:Preprocessing2U
Iterator::Model::ParallelMapV2M.??:??!??ߩ?@)M.??:??1??ߩ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?z????!\̚??E@)???x#??1?6???q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorx?캷"??!Y????h@)x?캷"??1Y????h@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice^??I?Ԁ?!?a75?@)^??I?Ԁ?1?a75?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipН`?un??!?Yg??2@)Y??L/1v?1Ԥ?p&???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9'ЮI2C??I0Q?ͼ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N??oD???N??oD???!N??oD???      ??!       "      ??!       *      ??!       2	~?D??T@~?D??T@!~?D??T@:      ??!       B      ??!       J	?~?nض???~?nض??!?~?nض??R      ??!       Z	?~?nض???~?nض??!?~?nض??b      ??!       JCPU_ONLYY'ЮI2C??b q0Q?ͼ?X@