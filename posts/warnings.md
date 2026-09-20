---
title: "AI coder warings lists"
date: 2025-11-26
description: ""
categories: ["warnings"]
author: "김한울"
---

```
/tmp/ipykernel_1028715/3025202350.py:20: FutureWarning: The return type of `Dataset.dims` will be changed to return a set of dimension names in future, in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names to lengths, please use `Dataset.sizes`.
  n_we = ds.dims['west_east']

/tmp/ipykernel_1028715/3025202350.py:21: FutureWarning: The return type of `Dataset.dims` will be changed to return a set of dimension names in future, in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names to lengths, please use `Dataset.sizes`.
  n_sn = ds.dims['south_north']

/tmp/ipykernel_1028715/1320644537.py:16: FutureWarning: The return type of `Dataset.dims` will be changed to return a set of dimension names in future, in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names to lengths, please use `Dataset.sizes`.
  for dim, size in ds.dims.items():

/tmp/ipykernel_1059231/2640257274.py:14: UserWarning: The specified chunks separate the stored chunks along dimension "south_north" starting at index 120. This could degrade performance. Instead, consider rechunking after loading.
  ds_u = xr.open_dataset(file_pairs[m]['u'], decode_times=False, chunks=CHUNKS)
/tmp/ipykernel_1059231/2640257274.py:14: UserWarning: The specified chunks separate the stored chunks along dimension "west_east" starting at index 145. This could degrade performance. Instead, consider rechunking after loading.
  ds_u = xr.open_dataset(file_pairs[m]['u'], decode_times=False, chunks=CHUNKS)
/tmp/ipykernel_1059231/2640257274.py:15: UserWarning: The specified chunks separate the stored chunks along dimension "south_north" starting at index 120. This could degrade performance. Instead, consider rechunking after loading.
  ds_v = xr.open_dataset(file_pairs[m]['v'], decode_times=False, chunks=CHUNKS)
/tmp/ipykernel_1059231/2640257274.py:15: UserWarning: The specified chunks separate the stored chunks along dimension "west_east" starting at index 145. This could degrade performance. Instead, consider rechunking after loading.
  ds_v = xr.open_dataset(file_pairs[m]['v'], decode_times=False, chunks=CHUNKS)
/tmp/ipykernel_1059231/2640257274.py:28: FutureWarning: The return type of `Dataset.dims` will be changed to return a set of dimension names in future, in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names to lengths, please use `Dataset.sizes`.
  'timesteps': ds_u.dims['Times'],

/tmp/ipykernel_1059231/1996839350.py:69: UserWarning: The specified chunks separate the stored chunks along dimension "south_north" starting at index 120. This could degrade performance. Instead, consider rechunking after loading.
  ds_u = xr.open_dataset(file_pairs[m]['u'], decode_times=False, chunks=CHUNKS)
/tmp/ipykernel_1059231/1996839350.py:69: UserWarning: The specified chunks separate the stored chunks along dimension "west_east" starting at index 145. This could degrade performance. Instead, consider rechunking after loading.
  ds_u = xr.open_dataset(file_pairs[m]['u'], decode_times=False, chunks=CHUNKS)
/tmp/ipykernel_1059231/1996839350.py:70: UserWarning: The specified chunks separate the stored chunks along dimension "south_north" starting at index 120. This could degrade performance. Instead, consider rechunking after loading.
  ds_v = xr.open_dataset(file_pairs[m]['v'], decode_times=False, chunks=CHUNKS)
/tmp/ipykernel_1059231/1996839350.py:70: UserWarning: The specified chunks separate the stored chunks along dimension "west_east" starting at index 145. This could degrade performance. Instead, consider rechunking after loading.
  ds_v = xr.open_dataset(file_pairs[m]['v'], decode_times=False, chunks=CHUNKS)
```
