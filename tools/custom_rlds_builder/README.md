# rlds 数据集创建说明

这里的类名就是数据集保存名称

## Quickstart

第一步：生成 rlds 支持格式的 hdf5

```bash
cd custom_rlds_builder
python simple_2_hdf5.py
```

第二步：将 hdf5 转为 rlds

```bash
cd custom_rlds_builder
tfds build
```

## faq

### `simple_2_hdf5.py` 脚本可能存在问题

这个脚本有时会导致生成失败，生成一个 莫名其妙的文件，暂时未找到问题！！！

### GPU ERROR

```
In <Dataset> with name "steps":
In <FeaturesDict> with name "observation":
InternalError: In <Image> with name "wrist_image":
Failed call to cuDeviceGet: CUDA_ERROR_NOT_INITIALIZED: initialization error
```

tensorflow gpu driver error

**use cpu**

```
CUDA_VISIBLE_DEVICES=-1
```

### network error

```
All attempts to get a Google authentication bearer token failed, returning an empty token. Retrieving token from files failed with "NOT_FOUND: Could not locate the credentials file.". Retrieving token from GCE failed with "FAILED_PRECONDITION: Error executing an HTTP request: libcurl code 6 meaning 'Couldn't resolve host name', error details: Could not resolve host: metadata.google.internal".
```

**代理**

!!!proxy!!!
