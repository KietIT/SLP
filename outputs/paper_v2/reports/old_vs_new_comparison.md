# So sánh output legacy và paper-v2

- Chế độ: `formal_fail_closed`
- Tổng số dòng: 797; có delta hợp lệ: 54.
- Comparable: 54; structural/không comparable: 743.

## Quy tắc diễn giải

VIVOS legacy (300 source utterance đã exposed) và paper-v2 (460 source utterance locked-unseen) không cùng scope. CSV vẫn giữ hai giá trị để audit, nhưng cố ý để trống `delta_new_minus_old`. FLEURS chỉ có delta khi exact ordered `utt_id/ref` của cả prediction cũ, prediction mới và manifest 857 câu khớp nhau.

## Thay đổi protocol chính

| Thuộc tính | Cũ | Mới | Kết luận |
| --- | --- | --- | --- |
| benchmark_observation_count | 1500 | 2300 | both use clean plus four SNR conditions, but their source utterances and noise partitions differ |
| lambda_selection_scope | same 300-source exposed benchmark used for lambda screening | held-out noisy-dev (14125 rows) before final-test unlock | the old selected checkpoint and new decision-locked checkpoint follow different selection protocols |
| noise_partition_protocol | legacy MUSAN pool without train/dev/test content lock | SHA-locked, content-disjoint MUSAN train/dev/test partitions | noise realizations are not exchangeable across protocol versions |
| source_utterance_count | 300 | 460 | 300 legacy-exposed and 460 locked-unseen utterances are disjoint; no metric delta is valid |
| training_pool_protocol | official_train+dev (11,660; dev included in historical fit pool) | official_train only (8,835); dev 2,825 held out | paper-v2 removes the historical dev-in-training validity blocker |

## FLEURS: các delta được phép

| Run | Metric/statistic | Cũ | Mới | Delta mới-cũ | Mức so sánh |
| --- | --- | ---: | ---: | ---: | --- |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | cer/ci_lower | -0.000636662936 | -0.002386225636 | -0.0017495627 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | cer/ci_upper | 0.003668238746 | 0.002127858018 | -0.001540380728 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | cer/delta_b_minus_a | 0.001602885193 | -0.000069690661 | -0.001672575854 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | der/ci_lower | -0.000297176690 | -0.001321113573 | -0.001023936883 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | der/ci_upper | 0.000789788148 | 0.000179423340 | -0.000610364808 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | der/delta_b_minus_a | 0.000220194383 | -0.000552749700 | -0.000772944083 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | ter/ci_lower | -0.002846655712 | -0.002864579401 | -1.79236889999998e-05 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | ter/ci_upper | 0.000953123805 | 0.000962990498 | 9.86669300000003e-06 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | ter/delta_b_minus_a | -0.000535864530 | -0.000891768592 | -0.000355904062 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | wer/ci_lower | 0.001936823541 | -0.001425442046 | -0.003362265587 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | wer/ci_upper | 0.007445150353 | 0.004581770935 | -0.002863379418 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | wer/delta_b_minus_a | 0.004738423607 | 0.001617998305 | -0.003120425302 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | cer/ci_lower | 0.000191845353 | -0.001642511436 | -0.001834356789 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | cer/ci_upper | 0.003518294187 | 0.001597614216 | -0.001920679971 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | cer/delta_b_minus_a | 0.001533194533 | 0.000008711333 | -0.0015244832 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | der/ci_lower | -0.000590165265 | -0.001282926246 | -0.000692760981 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | der/ci_upper | 0.000441447518 | 0.000006315637 | -0.000435131881 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | der/delta_b_minus_a | -0.000077541524 | -0.000598162096 | -0.000520620572 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | ter/ci_lower | -0.000717072920 | -0.001464749183 | -0.000747676263 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | ter/ci_upper | 0.002180880113 | 0.000429295160 | -0.001751584953 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | ter/delta_b_minus_a | 0.000441898051 | -0.000469546739 | -0.00091144479 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | wer/ci_lower | 0.001069156255 | -0.001581947669 | -0.002651103924 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | wer/ci_upper | 0.004775007125 | 0.003380102232 | -0.001394904893 | comparable_with_bootstrap_method_note |
| ordinary_baseline[ordinary_lora,lambda=0,seed=42]->selected_method[tone_aware_lora,lambda=0.05,seed=42] | wer/delta_b_minus_a | 0.002850758918 | 0.000886046691 | -0.001964712227 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | cer/ci_lower | -0.002216669959 | -0.002030455645 | 0.000186214314 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | cer/ci_upper | 0.002163951622 | 0.001561682901 | -0.000602268721 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | cer/delta_b_minus_a | 0.000069690661 | -0.000078401993 | -0.000148092654 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | der/ci_lower | -0.000162846780 | -0.000346275142 | -0.000183428362 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | der/ci_upper | 0.000773362262 | 0.000462843651 | -0.000310518611 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | der/delta_b_minus_a | 0.000297735907 | 0.000045412396 | -0.000252323511 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | ter/ci_lower | -0.003517110621 | -0.002264985586 | 0.001252125035 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | ter/ci_upper | 0.000850413360 | 0.001123559165 | 0.000273145805 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | ter/delta_b_minus_a | -0.000977762581 | -0.000422221853 | 0.000555540728 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | wer/ci_lower | -0.000528558709 | -0.001222536301 | -0.000693977592 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | wer/ci_upper | 0.004293690254 | 0.002745660501 | -0.001548029753 | comparable_with_bootstrap_method_note |
| selected_method[tone_aware_lora,lambda=0.05,seed=42]->locked_control[tone_aware_lora,lambda=0.1,seed=42] | wer/delta_b_minus_a | 0.001887664689 | 0.000731951614 | -0.001155713075 | comparable_with_bootstrap_method_note |
| role=locked_control;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.1;seed=42 | cer/rate | 0.10013676792138894 | 0.09935274798986 | -0.000784019931528934 | comparable_same_utterances |
| role=locked_control;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.1;seed=42 | der/rate | 0.005592841163310962 | 0.00520005157075938 | -0.000392789592551583 | comparable_same_utterances |
| role=locked_control;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.1;seed=42 | fcer/rate | 0.08518855065879145 | 0.08482142857142858 | -0.000367122087362878 | comparable_same_utterances |
| role=locked_control;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.1;seed=42 | swdr/rate | 0.004915346805024577 | 0.004915346805024577 | 0 | comparable_same_utterances |
| role=locked_control;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.1;seed=42 | ter/rate | 0.010683668633955111 | 0.009763440860215054 | -0.000920227773740057 | comparable_same_utterances |
| role=locked_control;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.1;seed=42 | wer/rate | 0.17112258263348487 | 0.16923491794437168 | -0.00188766468911319 | comparable_same_utterances |
| role=ordinary_baseline;dataset=fleurs;model=phowhisper;model_size=base;train_type=ordinary_lora;lambda=0;seed=42 | cer/rate | 0.09853388272804091 | 0.09942243865044036 | 0.000888555922399448 | comparable_same_utterances |
| role=ordinary_baseline;dataset=fleurs;model=phowhisper;model_size=base;train_type=ordinary_lora;lambda=0;seed=42 | der/rate | 0.005372646780710049 | 0.005752801270768042 | 0.000380154490057993 | comparable_same_utterances |
| role=ordinary_baseline;dataset=fleurs;model=phowhisper;model_size=base;train_type=ordinary_lora;lambda=0;seed=42 | fcer/rate | 0.0838255338482508 | 0.08373115101917103 | -9.43828290797671e-05 | comparable_same_utterances |
| role=ordinary_baseline;dataset=fleurs;model=phowhisper;model_size=base;train_type=ordinary_lora;lambda=0;seed=42 | swdr/rate | 0.004915346805024577 | 0.006007646095030038 | 0.00109229929000546 | comparable_same_utterances |
| role=ordinary_baseline;dataset=fleurs;model=phowhisper;model_size=base;train_type=ordinary_lora;lambda=0;seed=42 | ter/rate | 0.011219533164252246 | 0.010655209452201933 | -0.000564323712050313 | comparable_same_utterances |
| role=ordinary_baseline;dataset=fleurs;model=phowhisper;model_size=base;train_type=ordinary_lora;lambda=0;seed=42 | wer/rate | 0.1663841590261191 | 0.16761691963941752 | 0.00123276061329841 | comparable_same_utterances |
| role=selected_method;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.05;seed=42 | cer/rate | 0.10006707726080859 | 0.0994311499830129 | -0.000635927277795684 | comparable_same_utterances |
| role=selected_method;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.05;seed=42 | der/rate | 0.005295105256360583 | 0.005154639175257732 | -0.000140466081102852 | comparable_same_utterances |
| role=selected_method;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.05;seed=42 | fcer/rate | 0.08588724933787363 | 0.0849163702414289 | -0.000970879096444735 | comparable_same_utterances |
| role=selected_method;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.05;seed=42 | swdr/rate | 0.006553795740032769 | 0.006553795740032769 | 0 | comparable_same_utterances |
| role=selected_method;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.05;seed=42 | ter/rate | 0.011661431214768278 | 0.010185662712738524 | -0.00147576850202975 | comparable_same_utterances |
| role=selected_method;dataset=fleurs;model=phowhisper;model_size=base;train_type=tone_aware_lora;lambda=0.05;seed=42 | wer/rate | 0.16923491794437168 | 0.16850296633022574 | -0.000731951614145937 | comparable_same_utterances |

## Inventory đầu vào

| Artifact | Trạng thái | Số dòng | SHA-256 |
| --- | --- | ---: | --- |
| decision_lock | present |  | ccaeadc332fb38449fc6063143b190b1498e49dd2d6e39dd471a2174ab5aca8d |
| environment_lock | present |  | 7d7ffb0b09168d2ec32c325385d5359ea7134b1bc61de5f9b52c0224a25b8a4d |
| final_benchmark_lock | present |  | b34d7b25c320752b8d857a2f7749ad12ffafdf58cdd4a19efd76fdb8985d382d |
| fleurs_manifest | present | 857 | 91e6256c34faf7d26ed459844bd6a3dc390337716d67801f0eca80dcc72bcede |
| fleurs_preparation_lock | present |  | 19bca9ae70b19c7bf74dc8fdc1439d58142b5ebb9166590f4dbf92d8f5ee161e |
| method_lock | present |  | 0e20238c9b027d75937e4c65644394452990422ceab5f922a57be87c21f9007d |
| new_benchmark_manifest | present | 2300 | 89712f4873a6de7fb84f912d2380419fc48f56c7499ebed5f62528c749a2a146 |
| new_by_noise_type | present | 36 | 65f1bc32cc0acf2139e031da38881a641be0b043c975ce9d0303ca3ec1ca582c |
| new_by_snr | present | 63 | ee80853e084822daeb9f38871c5d2a5992df205f920e689547e71b7858542010 |
| new_final_bootstrap | present | 12 | 92d6e436bb19148900cf54f3613e57c16903ab51b7a433e54038b988cc8b9d7e |
| new_fleurs_bootstrap | present | 12 | 4118ad56506737c3bc71dc83879c2894240bf9f99388e3fa08f65529d57ccc54 |
| new_fleurs_provenance | present |  | bc3a83bf76b297233fbc3537c5d64f638a95ab83d619f610c85fa68b1f529a68 |
| new_fleurs_results | present | 3 | 44bd50a5ffc33eb15707b0eeb25df9deee642d0d8f7e936876d7f3beb7a98c31 |
| noise_split_lock | present |  | 5ebb46494c5613e586b9c9f93cffb8afbcbd4cdec21d198d02f198a1ee764cc9 |
| noisy_dev_lock | present |  | db263c9dc565a498261a1f912d7389ad4c947015d1b03f12befb89618417cda7 |
| old_benchmark_manifest | present | 1500 | a70139c89cbd123b8d0646588b4c7e3dc8c26abf0559cdac6772337d6d1bcfa6 |
| old_by_noise_type | present | 44 | 8eb56fe7be9e58e0816414d7340608453f4b6cad4f732e9e523a95b2bf9c85c5 |
| old_by_snr | present | 77 | a2f50bae2bdd2e09b1828b1fe6c1aacda860db8b596cb07fa47b088a4938bf73 |
| old_fleurs_bootstrap | present | 12 | d22c815cf5e5279683c03e14b4e182586f56ef20ec74aacdb68a92f50846df31 |
| old_fleurs_results | present | 3 | 1b39c2d0134a93d68fa1a38fdb96b58bd193c100b0923279108e7f5a649b0a10 |
| split_lock | present |  | a2b186e8f2b4d65bfa4ce5ac3511e23a9e000296e242dd12962c6fcfffc20de2 |
