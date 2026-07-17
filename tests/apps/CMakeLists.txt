add_executable(leakflow_cli_syntax_tests
    cli_syntax_test.cpp
)
target_link_libraries(leakflow_cli_syntax_tests
    PRIVATE
        leakflow_cli
)
leakflow_configure_test_target(leakflow_cli_syntax_tests)
add_test(NAME leakflow_cli_syntax COMMAND leakflow_cli_syntax_tests)

set(LEAKFLOW_AES_HDF5 "${PROJECT_SOURCE_DIR}/tests/fixtures/aes/sync/key_01.h5")

add_test(NAME leakflow_cli_help COMMAND leakflow --help)
set_tests_properties(leakflow_cli_help PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage: leakflow"
)

add_test(NAME leakflow_cli_pipeline COMMAND leakflow --pipeline fake-src-summary)
set_tests_properties(leakflow_cli_pipeline PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_set_property
    COMMAND leakflow --pipeline fake-src-summary --set fake_src.caps_type=sca/test
)
set_tests_properties(leakflow_cli_set_property PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/test"
)

add_test(NAME leakflow_cli_tee_pipeline COMMAND leakflow --pipeline fake-src-tee-summary-sink)
set_tests_properties(leakflow_cli_tee_pipeline PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_simple COMMAND leakflow run "FakeSrc ! Summary")
set_tests_properties(leakflow_cli_run_simple PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_telemetry_option COMMAND leakflow run --telemetry "FakeSrc ! Summary")
set_tests_properties(leakflow_cli_run_telemetry_option PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_no_telemetry_option COMMAND leakflow run --no-telemetry "FakeSrc ! Summary")
set_tests_properties(leakflow_cli_run_no_telemetry_option PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_no_summaries_forced
    COMMAND leakflow --no-summaries run "FakeSrc ! Summary(always_print=true)"
)
set_tests_properties(leakflow_cli_run_no_summaries_forced PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_log_filter
    COMMAND leakflow --log-level info --log-color never --log-filter element=Hdf5FileSrc run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! FakeSink"
)
set_tests_properties(leakflow_cli_run_log_filter PROPERTIES
    PASS_REGULAR_EXPRESSION "Hdf5FileSrc@data.traces -> FakeSink"
)

add_test(NAME leakflow_cli_run_env_log_level
    COMMAND leakflow run "FakeSrc ! FakeSink"
)
set_tests_properties(leakflow_cli_run_env_log_level PROPERTIES
    ENVIRONMENT "LEAKFLOW_LOG_LEVEL=debug;LEAKFLOW_LOG_COLOR=never;LEAKFLOW_LOG_FILTER=element=FakeSrc"
    PASS_REGULAR_EXPRESSION "element=FakeSrc"
)

add_test(NAME leakflow_cli_summary_level_full
    COMMAND leakflow --log-level info --log-color never --summary-level 2 run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! FakeSink"
)
set_tests_properties(leakflow_cli_summary_level_full PROPERTIES
    PASS_REGULAR_EXPRESSION "contiguous="
)

add_test(NAME leakflow_cli_summary_level_terse
    COMMAND leakflow --log-level info --log-color never --summary-level 1 run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! FakeSink"
)
set_tests_properties(leakflow_cli_summary_level_terse PROPERTIES
    PASS_REGULAR_EXPRESSION "shape="
    FAIL_REGULAR_EXPRESSION "contiguous="
)

add_test(NAME leakflow_cli_summary_level_env
    COMMAND leakflow --log-level info --log-color never run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! FakeSink"
)
set_tests_properties(leakflow_cli_summary_level_env PROPERTIES
    ENVIRONMENT "LEAKFLOW_SUMMARY_LEVEL=1"
    PASS_REGULAR_EXPRESSION "shape="
    FAIL_REGULAR_EXPRESSION "contiguous="
)

add_test(NAME leakflow_cli_log_payload_nested_fields
    COMMAND leakflow --log-level info --log-color never run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); Tee@traces; AesLeakage@leakage(units=[3]); PearsonCorrelator@corr; PoiSelect@poi(top_k=[2],rank_by=[abs]); CorrelationPoiToPlotAnnotations@ann(precision=3); FakeSink@sink; @data.traces ! @traces; @traces.src_0 ! @corr.features; @traces.src_1 ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @corr.targets; @corr ! @poi; @poi ! @ann ! @sink"
)
set_tests_properties(leakflow_cli_log_payload_nested_fields PROPERTIES
    PASS_REGULAR_EXPRESSION "sample_index="
)

add_test(NAME leakflow_cli_run_properties
    COMMAND leakflow run "FakeSrc(caps_type=sca/test, metadata_value=\"hello leakflow\") ! Summary(level=3)"
)
set_tests_properties(leakflow_cli_run_properties PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/test.*source=hello leakflow"
)

add_test(NAME leakflow_cli_run_named_declarations
    COMMAND leakflow run "FakeSrc@src; Summary@s(level=2); @src(caps_type=sca/named); @src ! @s"
)
set_tests_properties(leakflow_cli_run_named_declarations PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/named"
)

add_test(NAME leakflow_cli_run_tee
    COMMAND leakflow run "FakeSrc ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! FakeSink"
)
set_tests_properties(leakflow_cli_run_tee PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_normalized_names
    COMMAND leakflow run "fake-src ! summary"
)
set_tests_properties(leakflow_cli_run_normalized_names PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_queue_properties
    COMMAND leakflow run "FakeSrc ! Queue(max_size=1, drop_oldest=false) ! Summary"
)
set_tests_properties(leakflow_cli_run_queue_properties PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_caps_annotation
    COMMAND leakflow run "FakeSrc@src; @src.src[caps=sca/fake; dtype=float32] ! Summary@s"
)
set_tests_properties(leakflow_cli_run_caps_annotation PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_element_caps_annotation
    COMMAND leakflow run "FakeSrc[caps=sca/fake] ! Summary"
)
set_tests_properties(leakflow_cli_run_element_caps_annotation PROPERTIES
    PASS_REGULAR_EXPRESSION "caps=sca/fake"
)

add_test(NAME leakflow_cli_run_metadata_annotation
    COMMAND leakflow run "FakeSrc{dataset=smoke; note=\"hello leakflow\"} ! Summary(level=3)"
)
set_tests_properties(leakflow_cli_run_metadata_annotation PROPERTIES
    PASS_REGULAR_EXPRESSION "dataset=smoke"
)

add_test(NAME leakflow_cli_run_combined_annotations
    COMMAND leakflow run "FakeSrc(caps_type=sca/test)[caps=sca/fake; dtype=float32]{dataset=smoke; note=\"combined\"} ! Summary(level=3)"
)
set_tests_properties(leakflow_cli_run_combined_annotations PROPERTIES
    PASS_REGULAR_EXPRESSION "dataset=smoke"
)

add_test(NAME leakflow_cli_run_tee_metadata
    COMMAND leakflow run "FakeSrc ! Tee@t{dataset=smoke}; @t.src_0{branch=summary} ! Summary(level=3); @t.src_1{branch=sink} ! FakeSink"
)
set_tests_properties(leakflow_cli_run_tee_metadata PROPERTIES
    PASS_REGULAR_EXPRESSION "branch=summary"
)

add_test(NAME leakflow_cli_run_tee_template_metadata
    COMMAND leakflow run "FakeSrc ! Tee@t; @t.src_%u{branch=template}; @t.src_0{branch=summary} ! Summary(level=3); @t.src_1 ! Summary(level=3)"
)
set_tests_properties(leakflow_cli_run_tee_template_metadata PROPERTIES
    PASS_REGULAR_EXPRESSION "branch=template"
)

add_test(NAME leakflow_cli_run_file_src_summary
    COMMAND leakflow run "FileSrc(path=/dev/null) ! Summary"
)
set_tests_properties(leakflow_cli_run_file_src_summary PROPERTIES
    PASS_REGULAR_EXPRESSION "payload=leakflow/bytes"
)

add_test(NAME leakflow_cli_run_torch_file_src_summary_sink
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! Summary ! FakeSink"
)
set_tests_properties(leakflow_cli_run_torch_file_src_summary_sink PROPERTIES
    PASS_REGULAR_EXPRESSION "Buffer@summary"
)

add_test(NAME leakflow_cli_run_torch_file_src_tensor_summary_sink
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! Summary ! FakeSink"
)
set_tests_properties(leakflow_cli_run_torch_file_src_tensor_summary_sink PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow/torch-tensor"
)

add_test(NAME leakflow_cli_run_torch_convert_summary_sink
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! TorchConvert(dtype=float32,device=cpu) ! Summary(level=3) ! FakeSink"
)
set_tests_properties(leakflow_cli_run_torch_convert_summary_sink PROPERTIES
    PASS_REGULAR_EXPRESSION "conversion.id=torch-convert"
)

add_test(NAME leakflow_cli_run_aes_leakage
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); AesLeakage@leakage(channels=[HW(m)],units=[0,1]); Summary@summary(level=3); @data.traces ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @leakage ! @summary"
)
set_tests_properties(leakflow_cli_run_aes_leakage PROPERTIES
    PASS_REGULAR_EXPRESSION "payload.leakage.model=aes-first-round"
)

add_test(NAME leakflow_cli_run_aes_leakage_hypothesis
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); AesLeakageHypothesis@hyp(units=[0],guess_values=[0,255]); Summary@summary(level=3); @data.plaintexts ! @hyp.plaintexts; @hyp ! @summary"
)
set_tests_properties(leakflow_cli_run_aes_leakage_hypothesis PROPERTIES
    PASS_REGULAR_EXPRESSION "attack.guess.count=2"
)

add_test(NAME leakflow_cli_run_cpa_attack
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); AesLeakageHypothesis@hyp(units=[0],guess_values=[0,255]); CpaAttack@attack(compute_dtype=float64); Summary@summary(level=2); @data.traces ! @attack.features; @data.plaintexts ! @hyp.plaintexts; @hyp ! @attack.hypotheses; @attack ! @summary"
)
set_tests_properties(leakflow_cli_run_cpa_attack PROPERTIES
    PASS_REGULAR_EXPRESSION "payload=leakflow/attack-scores"
)

add_test(NAME leakflow_cli_run_attack_stats
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); AesLeakageHypothesis@hyp(units=[0],guess_values=[0,255]); CpaAttack@attack(compute_dtype=float64); AttackStats@stats; Summary@summary(level=2); @data.traces ! @attack.features; @data.plaintexts ! @hyp.plaintexts; @hyp ! @attack.hypotheses; @attack ! @stats.scores; @data.keys ! @stats.truth; @stats ! @summary"
)
set_tests_properties(leakflow_cli_run_attack_stats PROPERTIES
    PASS_REGULAR_EXPRESSION "payload.stats.id"
)

add_test(NAME leakflow_cli_run_attack_annotations
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); AesLeakageHypothesis@hyp(units=[0],guess_values=[0,255]); CpaAttack@attack(compute_dtype=float64); AttackStats@stats(top_k=2); AttackStatsToPlotAnnotations@ann(precision=3); Summary@summary(level=2); @data.traces ! @attack.features; @data.plaintexts ! @hyp.plaintexts; @hyp ! @attack.hypotheses; @attack ! @stats.scores; @data.keys ! @stats.truth; @stats ! @ann ! @summary"
)
set_tests_properties(leakflow_cli_run_attack_annotations PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow/plot-annotations"
)

add_test(NAME leakflow_cli_run_poi_select
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); Tee@traces; AesLeakage@leakage(units=[0]); PearsonCorrelator@corr; PoiSelect@poi(top_k=[3],rank_by=[abs]); Summary@summary(level=2); @data.traces ! @traces; @traces.src_0 ! @corr.features; @traces.src_1 ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @corr.targets; @corr ! @poi; @poi ! @summary"
)
set_tests_properties(leakflow_cli_run_poi_select PROPERTIES
    PASS_REGULAR_EXPRESSION "payload.poi.method"
)

add_test(NAME leakflow_cli_run_correlation_poi_to_plot_annotations
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); Tee@traces; AesLeakage@leakage(units=[3,5]); PearsonCorrelator@corr; PoiSelect@poi(top_k=[3],rank_by=[abs]); CorrelationPoiToPlotAnnotations@ann(precision=3); Summary@summary(level=2); @data.traces ! @traces; @traces.src_0 ! @corr.features; @traces.src_1 ! @leakage.traces; @data.plaintexts ! @leakage.plaintexts; @data.keys ! @leakage.keys; @leakage ! @corr.targets; @corr ! @poi; @poi ! @ann ! @summary"
)
set_tests_properties(leakflow_cli_run_correlation_poi_to_plot_annotations PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow/plot-annotations"
)

add_test(NAME leakflow_cli_run_clustering_evaluate
    COMMAND leakflow --log-level warning run
        "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); AesLeakage@leak(channels=[HW(m),HW(y)],units=[0]); Tee@targets; GaussianMixture@gmm(n_components=2,covariance_type=diagonal,n_init=1,max_iter=2,seed=0); ClusteringEvaluate@eval(semantic=power,semantic_ranges=[8,8],dimension_names=[hm,hy],alignment=both); Summary@summary(level=2); @data.traces ! @leak.traces; @data.plaintexts ! @leak.plaintexts; @data.keys ! @leak.keys; @leak ! @targets; @targets.src_0 ! @gmm ! @eval.labels; @targets.src_1 ! @eval.truth; @eval ! @summary"
)
set_tests_properties(leakflow_cli_run_clustering_evaluate PROPERTIES
    PASS_REGULAR_EXPRESSION "payload=leakflow/clustering-evaluation"
)

add_test(NAME leakflow_cli_run_torch_file_sink
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! TorchFileSink(path=${PROJECT_BINARY_DIR}/tests/traces_roundtrip.pt)"
)

add_test(NAME leakflow_cli_run_torch_file_sink_after_tee
    COMMAND leakflow run "Hdf5FileSrc@data(path=${LEAKFLOW_AES_HDF5}); @data.traces ! Tee@t; @t.src_0 ! Summary(level=2); @t.src_1 ! TorchFileSink(path=${PROJECT_BINARY_DIR}/tests/traces_after_tee.pt)"
)
set_tests_properties(leakflow_cli_run_torch_file_sink_after_tee PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow/torch-tensor"
)

add_test(NAME leakflow_cli_run_ambiguous_tee COMMAND leakflow run "FakeSrc ! Tee@t; @t ! Summary")
set_tests_properties(leakflow_cli_run_ambiguous_tee PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_missing_pad COMMAND leakflow run "FakeSrc@src; @src.missing ! Summary")
set_tests_properties(leakflow_cli_run_missing_pad PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_unknown_element COMMAND leakflow run "MissingElement ! Summary")
set_tests_properties(leakflow_cli_run_unknown_element PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_duplicate_name COMMAND leakflow run "FakeSrc@x; Summary@x")
set_tests_properties(leakflow_cli_run_duplicate_name PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_bad_property COMMAND leakflow run "Summary(level=9)")
set_tests_properties(leakflow_cli_run_bad_property PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_bad_caps_annotation COMMAND leakflow run "FakeSrc@src; @src.src[dtype=float32]")
set_tests_properties(leakflow_cli_run_bad_caps_annotation PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_bad_metadata_annotation COMMAND leakflow run "FakeSrc@src; @src.src{dataset}")
set_tests_properties(leakflow_cli_run_bad_metadata_annotation PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_ambiguous_caps_annotation
    COMMAND leakflow run "FakeSrc ! Tee@t; @t[caps=sca/fake] ! Summary"
)
set_tests_properties(leakflow_cli_run_ambiguous_caps_annotation PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_unmatched_metadata_template
    COMMAND leakflow run "FakeSrc ! Tee@t; @t.out_%u{branch=summary}"
)
set_tests_properties(leakflow_cli_run_unmatched_metadata_template PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_cli_run_metadata_template_as_link
    COMMAND leakflow run "FakeSrc ! Tee@t; @t.src_%u ! Summary"
)
set_tests_properties(leakflow_cli_run_metadata_template_as_link PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME leakflow_ls_list COMMAND leakflow-ls)
set_tests_properties(leakflow_ls_list PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow_plugins_core"
)

add_test(NAME leakflow_ls_list_extras COMMAND leakflow-ls)
set_tests_properties(leakflow_ls_list_extras PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow_plugins_extras.*NumpySrc.*NumpyToTorch.*Hdf5FileSrc.*FakeLiveHdf5Src"
)

add_test(NAME leakflow_ls_list_plot COMMAND leakflow-ls)
set_tests_properties(leakflow_ls_list_plot PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow_plugins_plot.*TracePlot"
)

add_test(NAME leakflow_ls_list_crypto COMMAND leakflow-ls)
set_tests_properties(leakflow_ls_list_crypto PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow_plugins_crypto.*AesLeakage"
)

add_test(NAME leakflow_ls_trace_plot COMMAND leakflow-ls TracePlot)
set_tests_properties(leakflow_ls_trace_plot PROPERTIES
    PASS_REGULAR_EXPRESSION "sample_rate_hz"
)

add_test(NAME leakflow_ls_aes_leakage COMMAND leakflow-ls AesLeakage)
set_tests_properties(leakflow_ls_aes_leakage PROPERTIES
    PASS_REGULAR_EXPRESSION "HW\\(m_xor_k\\)"
)

add_test(NAME leakflow_ls_aes_leakage_hypothesis COMMAND leakflow-ls AesLeakageHypothesis)
set_tests_properties(leakflow_ls_aes_leakage_hypothesis PROPERTIES
    PASS_REGULAR_EXPRESSION "guess_values"
)

add_test(NAME leakflow_ls_cpa_attack COMMAND leakflow-ls CpaAttack)
set_tests_properties(leakflow_ls_cpa_attack PROPERTIES
    PASS_REGULAR_EXPRESSION "score_method"
)

add_test(NAME leakflow_ls_attack_stats COMMAND leakflow-ls AttackStats)
set_tests_properties(leakflow_ls_attack_stats PROPERTIES
    PASS_REGULAR_EXPRESSION "true_rank"
)

add_test(NAME leakflow_ls_clustering_evaluate COMMAND leakflow-ls --no-colors ClusteringEvaluate)
set_tests_properties(leakflow_ls_clustering_evaluate PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow/clustering-evaluation"
)

add_test(NAME leakflow_ls_clustering_metrics_table_plot
    COMMAND leakflow-ls --no-colors ClusteringMetricsTablePlot)
set_tests_properties(leakflow_ls_clustering_metrics_table_plot PROPERTIES
    PASS_REGULAR_EXPRESSION "update_mode"
)

add_test(NAME leakflow_ls_list_ml_plot COMMAND leakflow-ls)
set_tests_properties(leakflow_ls_list_ml_plot PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow_plugins_ml_plot.*ClusteringMetricsTablePlot"
)

add_test(NAME leakflow_ls_attack_stats_to_plot_annotations COMMAND leakflow-ls AttackStatsToPlotAnnotations)
set_tests_properties(leakflow_ls_attack_stats_to_plot_annotations PROPERTIES
    PASS_REGULAR_EXPRESSION "plot annotations"
)

add_test(NAME leakflow_ls_poi_select COMMAND leakflow-ls PoiSelect)
set_tests_properties(leakflow_ls_poi_select PROPERTIES
    PASS_REGULAR_EXPRESSION "rank_by"
)

add_test(NAME leakflow_ls_correlation_poi_to_plot_annotations COMMAND leakflow-ls CorrelationPoiToPlotAnnotations)
set_tests_properties(leakflow_ls_correlation_poi_to_plot_annotations PROPERTIES
    PASS_REGULAR_EXPRESSION "plot annotation"
)

add_test(NAME leakflow_ls_list_base COMMAND leakflow-ls)
set_tests_properties(leakflow_ls_list_base PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow_plugins_base.*TorchFileSrc.*TorchFileSink"
)

add_test(NAME leakflow_ls_inspect COMMAND leakflow-ls leakflow_plugins_core)
set_tests_properties(leakflow_ls_inspect PROPERTIES
    PASS_REGULAR_EXPRESSION "Apache-2.0"
)

add_test(NAME leakflow_ls_inspect_base COMMAND leakflow-ls leakflow_plugins_base)
set_tests_properties(leakflow_ls_inspect_base PROPERTIES
    PASS_REGULAR_EXPRESSION "libleakflow_plugins_base.so"
)

add_test(NAME leakflow_ls_inspect_base_description COMMAND leakflow-ls --no-colors leakflow_plugins_base)
set_tests_properties(leakflow_ls_inspect_base_description PROPERTIES
    PASS_REGULAR_EXPRESSION "Description[ ]+Leakflow base elements"
)

add_test(NAME leakflow_ls_inspect_base_version COMMAND leakflow-ls --no-colors leakflow_plugins_base)
set_tests_properties(leakflow_ls_inspect_base_version PROPERTIES
    PASS_REGULAR_EXPRESSION "Version[ ]+0[.]10"
)

add_test(NAME leakflow_ls_inspect_extras COMMAND leakflow-ls leakflow_plugins_extras)
set_tests_properties(leakflow_ls_inspect_extras PROPERTIES
    PASS_REGULAR_EXPRESSION "libleakflow_plugins_extras.so"
)

add_test(NAME leakflow_ls_plugin_feature_count_leaf COMMAND leakflow-ls --no-colors --ascii leakflow_plugins_core)
set_tests_properties(leakflow_ls_plugin_feature_count_leaf PROPERTIES
    PASS_REGULAR_EXPRESSION "  `- 8 elements"
)

add_test(NAME leakflow_ls_inspect_element COMMAND leakflow-ls FakeSink)
set_tests_properties(leakflow_ls_inspect_element PROPERTIES
    PASS_REGULAR_EXPRESSION "Zer0Leak"
)

add_test(NAME leakflow_ls_inspect_tee_metadata_template COMMAND leakflow-ls --no-colors Tee)
set_tests_properties(leakflow_ls_inspect_tee_metadata_template PROPERTIES
    PASS_REGULAR_EXPRESSION "Output pad template: src_%u"
)

add_test(NAME leakflow_ls_inspect_name_property COMMAND leakflow-ls --no-colors Tee)
set_tests_properties(leakflow_ls_inspect_name_property PROPERTIES
    PASS_REGULAR_EXPRESSION "name : The name of the element instance"
)

add_test(NAME leakflow_ls_inspect_name_default COMMAND leakflow-ls --no-colors Tee)
set_tests_properties(leakflow_ls_inspect_name_default PROPERTIES
    PASS_REGULAR_EXPRESSION "Default: tee0"
)

add_test(NAME leakflow_ls_inspect_queue_telemetry COMMAND leakflow-ls --no-colors Queue)
set_tests_properties(leakflow_ls_inspect_queue_telemetry PROPERTIES
    PASS_REGULAR_EXPRESSION "peak_size : highest queue depth observed in the current run"
)

add_test(NAME leakflow_ls_inspect_numpy_src COMMAND leakflow-ls NumpySrc)
set_tests_properties(leakflow_ls_inspect_numpy_src PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow/numpy-array"
)

add_test(NAME leakflow_ls_inspect_numpy_to_torch COMMAND leakflow-ls NumpyToTorch)
set_tests_properties(leakflow_ls_inspect_numpy_to_torch PROPERTIES
    PASS_REGULAR_EXPRESSION "numpy-to-torch"
)

add_test(NAME leakflow_ls_inspect_hdf5_file_src COMMAND leakflow-ls --no-colors Hdf5FileSrc)
set_tests_properties(leakflow_ls_inspect_hdf5_file_src PROPERTIES
    PASS_REGULAR_EXPRESSION "storage_read"
)

add_test(NAME leakflow_ls_inspect_fake_live_hdf5_src COMMAND leakflow-ls --no-colors FakeLiveHdf5Src)
set_tests_properties(leakflow_ls_inspect_fake_live_hdf5_src PROPERTIES
    PASS_REGULAR_EXPRESSION "batch_size"
)

add_test(NAME leakflow_ls_inspect_torch_file_sink COMMAND leakflow-ls TorchFileSink)
set_tests_properties(leakflow_ls_inspect_torch_file_sink PROPERTIES
    PASS_REGULAR_EXPRESSION "leakflow/torch-tensor"
)

add_test(NAME leakflow_ls_inspect_torch_file_src_rank COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_torch_file_src_rank PROPERTIES
    PASS_REGULAR_EXPRESSION "primary [(]256[)]"
)

add_test(NAME leakflow_ls_inspect_torch_file_src_long_name COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_torch_file_src_long_name PROPERTIES
    PASS_REGULAR_EXPRESSION "Long-name[ ]+Torch File Source"
)

add_test(NAME leakflow_ls_inspect_torch_file_src_klass COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_torch_file_src_klass PROPERTIES
    PASS_REGULAR_EXPRESSION "Klass[ ]+Source/File/Torch"
)

add_test(NAME leakflow_ls_inspect_torch_convert_enum_values COMMAND leakflow-ls --no-colors TorchConvert)
set_tests_properties(leakflow_ls_inspect_torch_convert_enum_values PROPERTIES
    PASS_REGULAR_EXPRESSION "Allowed values: preserve, bool"
)

add_test(NAME leakflow_ls_inspect_torch_file_src_device_values COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_torch_file_src_device_values PROPERTIES
    PASS_REGULAR_EXPRESSION "Accepted values: .*cuda:0.*preserve"
)

add_test(NAME leakflow_ls_inspect_element_metadata COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_element_metadata PROPERTIES
    PASS_REGULAR_EXPRESSION "Element Metadata:"
)

add_test(NAME leakflow_ls_inspect_element_metadata_set COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_element_metadata_set PROPERTIES
    PASS_REGULAR_EXPRESSION "file[.]format"
)

add_test(NAME leakflow_ls_inspect_element_metadata_suggestions COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_element_metadata_suggestions PROPERTIES
    PASS_REGULAR_EXPRESSION "capture[.]source"
)

add_test(NAME leakflow_ls_inspect_element_metadata_type COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_element_metadata_type PROPERTIES
    PASS_REGULAR_EXPRESSION "type: double interval"
)

add_test(NAME leakflow_ls_inspect_element_metadata_examples COMMAND leakflow-ls --no-colors TorchFileSrc)
set_tests_properties(leakflow_ls_inspect_element_metadata_examples PROPERTIES
    PASS_REGULAR_EXPRESSION "Examples: .*-0[.]5,0[.]5"
)

add_test(NAME leakflow_ls_hierarchy_element_leaf COMMAND leakflow-ls --no-colors --ascii NumpySrc)
set_tests_properties(leakflow_ls_hierarchy_element_leaf PROPERTIES
    PASS_REGULAR_EXPRESSION "`- Element"
)

add_test(NAME leakflow_ls_hierarchy_child_spacing COMMAND leakflow-ls --no-colors --ascii NumpySrc)
set_tests_properties(leakflow_ls_hierarchy_child_spacing PROPERTIES
    PASS_REGULAR_EXPRESSION "   `- NumpySrc"
)

add_test(NAME leakflow_ls_no_colors COMMAND leakflow-ls --no-colors fakesink)
set_tests_properties(leakflow_ls_no_colors PROPERTIES
    PASS_REGULAR_EXPRESSION "Element Properties:"
)

add_test(NAME leakflow_ls_no_pager COMMAND leakflow-ls --no-pager --no-colors fakesink)
set_tests_properties(leakflow_ls_no_pager PROPERTIES
    PASS_REGULAR_EXPRESSION "Element Properties:"
)

add_test(NAME leakflow_ls_help_pager_environment COMMAND leakflow-ls --help)
set_tests_properties(leakflow_ls_help_pager_environment PROPERTIES
    PASS_REGULAR_EXPRESSION "LEAKFLOW_LS_LESS"
)

add_test(NAME leakflow_ls_unknown COMMAND leakflow-ls missing)
set_tests_properties(leakflow_ls_unknown PROPERTIES
    WILL_FAIL TRUE
)

if(TARGET leakflow_cuda_smoke)
    add_test(NAME leakflow_cuda_smoke COMMAND leakflow_cuda_smoke)
    set_tests_properties(leakflow_cuda_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "Vector add OK"
    )
endif()
