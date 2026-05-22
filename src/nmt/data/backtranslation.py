"""Back-translation augmentation: translate monolingual English with a trained reverse
en->hi model, prefix the synthetic source with the config.bt_tag token (tagged BT),
and merge the synthetic pairs into the forward hi->en training set."""
