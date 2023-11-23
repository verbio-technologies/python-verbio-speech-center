## To be released

### Features :tada:

### Bug fixes :bug:

### Removed :no_entry:

## 1.1.2 (2023-11-23)

### Patch (1 change)

- [Update version](csr/asr4-streaming@ebf1c88132d7f19b49507460f24dce9f7e87f651) ([merge request](csr/asr4-streaming!362))

## 1.0.6 (2023-11-13)

### Patch (5 changes)

- [Update w2v-engine versions to avoid conflicts with asr4-engine core](csr/asr4-streaming@e3b52a9827c8a1312d42a831bca926c416955666) ([merge request](csr/asr4-streaming!356))
- [[SC2-1152] Update W2V engine version to fix bugs](csr/asr4-streaming@54210b09a51801cf6f619a9a77a901a55353d1d5) ([merge request](csr/asr4-streaming!354))
- [Fix/Convert en-XX and es-XX language codes into w2v accepted language codes](csr/asr4-streaming@54d7c4344fb7afbf9fc26193d1530ef441aabfe9) ([merge request](csr/asr4-streaming!352))
- [[EEE-1141] Refactor/ Adjust timestamps and partial durations to total audio duration](csr/asr4-streaming@aa77d4a9db91ad4aeffed4e66a9c0e1d6aae3a3c) ([merge request](csr/asr4-streaming!348))
- [upgraded base image to 0.1.1](csr/asr4-streaming@e919739bee0ebc739cc48b11754d6a559b4158a0) ([merge request](csr/asr4-streaming!349))

## 1.0.5 (2023-11-09)

### Patch (4 changes)

- [[SC2-1152] Update W2V engine version to fix bugs](csr/asr4-streaming@54210b09a51801cf6f619a9a77a901a55353d1d5) ([merge request](csr/asr4-streaming!354))
- [Fix/Convert en-XX and es-XX language codes into w2v accepted language codes](csr/asr4-streaming@54d7c4344fb7afbf9fc26193d1530ef441aabfe9) ([merge request](csr/asr4-streaming!352))
- [[EEE-1141] Refactor/ Adjust timestamps and partial durations to total audio duration](csr/asr4-streaming@aa77d4a9db91ad4aeffed4e66a9c0e1d6aae3a3c) ([merge request](csr/asr4-streaming!348))
- [upgraded base image to 0.1.1](csr/asr4-streaming@e919739bee0ebc739cc48b11754d6a559b4158a0) ([merge request](csr/asr4-streaming!349))

## 1.0.4 (2023-10-11)

### Minor (1 change)

- [Update to version 1.4.0 of models](csr/asr4-streaming@7785d15df63de0e402c98c8b4eec4238262edced) ([merge request](csr/asr4-streaming!340))

### Patch (7 changes)

- [[Release 1.0.4] Fix argument processor mechanism in server.py and add test](csr/asr4-streaming@1060f0305ed2ef86954336ec4a2163d31e865187) ([merge request](csr/asr4-streaming!347))
- [Update asr4-engine to 0.10.2](csr/asr4-streaming@f30bc942e6cf7eedd0fb6ac471b7105a7cebe8c5) ([merge request](csr/asr4-streaming!346))
- [Changing all models to 1.0.3](csr/asr4-streaming@75e7dc795b9e596698fe4685e8df40a6cd74b740) ([merge request](csr/asr4-streaming!345))
- [[EEE-1051] Remove configuration that should be in the engine](csr/asr4-streaming@811a8f4338b28331a37ae6806cd8a98992787987) ([merge request](csr/asr4-streaming!343))
- [[SC2-1052] Refactor/ Close gRPC Channel on Internal Exception](csr/asr4-streaming@89d01899939776f7bddd72ffc7b33e81f388bb2f) ([merge request](csr/asr4-streaming!344))
- [Fix/ TRN File Generation](csr/asr4-streaming@9081c0286f600851fae00b9003bef7d531dc97d5) ([merge request](csr/asr4-streaming!342))
- [[EEE-1051] Remove unused arguments in server.py](csr/asr4-streaming@62c0715bd056c49313f5259133e451da45b99569) ([merge request](csr/asr4-streaming!341))

## 1.0.3 (2023-10-04)

### Patch (3 changes)

- [release/1.0.3](csr/asr4-streaming@66232d393766bedd6fceed61588d6024cf45693f) ([merge request](csr/asr4-streaming!337))
- [Fix/ Send Error Status Message before aborting a gRPC Channel](csr/asr4-streaming@fee6d15a950b8f44d716c5d9540466be8d94a421) ([merge request](csr/asr4-streaming!339))
- [[EEE-1050] Remove dependencies from W2V engine in client.py](csr/asr4-streaming@34904d10c9b04d38b796b6bba3cdddbebcbba4bf) ([merge request](csr/asr4-streaming!336))

## 1.0.0 (2023-09-15)

### Major (1 change)

- [[EEE-868] Feature/ Use asr4-engine Online Decoding API](csr/asr4@cebae65a774902bbcff830a37287294b7c571c4c) ([merge request](csr/asr4!302))

### Patch (7 changes)

- [release/1.0.0](csr/asr4@51f047a7d38605f4749496418f43c0d4a811ce60) ([merge request](csr/asr4!328))
- [[EEE-1085] Make client.py a bidirectional sender/reader](csr/asr4@7bc1f0adc1e064f782ffffbb64ea0292dc5e164c) ([merge request](csr/asr4!327))
- [Fix/ Streaming Functional Testing Latency Limit](csr/asr4@69eadc1761a292971d95ed14ceed389651a55079) ([merge request](csr/asr4!329))
- [[EEE-1049] Add loguru](csr/asr4@f086a3eefd6ff7aa40a0aea703ad1a8081904700) ([merge request](csr/asr4!324))
- [[EEE-945] Refactor/ gRPC Streaming API](csr/asr4@d758f34d21109f78cec24dcbe1101a1450de176f) ([merge request](csr/asr4!323))
- [Fix/ Code Coverage](csr/asr4@af243bd1a391dd780189c3bb19b34c3f65ea0ee2) ([merge request](csr/asr4!325))
- [[EEE-945] Refactor/ Remove gRPC Batch API](csr/asr4@ebb183ebea4c3fe5adfd6b1cfbe8ff9b6b10c165) ([merge request](csr/asr4!322))

## 0.7.8 (2023-09-04)

### Patch (9 changes)

- [Fix: Update release](csr/asr4@79aefbd9b8e9d6b0f9cfc4a74c24373c183adcd5) ([merge request](csr/asr4!321))
- [Increase grpc timeout from 3 to 4.](csr/asr4@9a6087350b2a5863f4c66e947faf54e1ac49b12d) ([merge request](csr/asr4!320))
- [Increase grpc timeout from 2 to 3](csr/asr4@8113df7cf98981c13e2f74c35476120cc59d0afc) ([merge request](csr/asr4!319))
- [Fix release 0.7.8](csr/asr4@0992924aebb9fe4179cb651b3dea810fcda74468) ([merge request](csr/asr4!318))
- [Fix some testing timeouts](csr/asr4@2234bda943ed50b66a702cc76d683d9da5f5dec5) ([merge request](csr/asr4!317))
- [Release 0.7.8](csr/asr4@b6a1c7afa54db9151e02744b5e7106f632f1aaa3) ([merge request](csr/asr4!316))
- [Fix/launch server ing gpu for integration tests](csr/asr4@a5864e1e12d717bfdc42b9d73044af40c45fd02c) ([merge request](csr/asr4!315))
- [Remove unused config file](csr/asr4@0f9419d8b00637087549537c9f24bf4b7b257a0c) ([merge request](csr/asr4!314))
- [[EEE-1044] fix streaming behaviour](csr/asr4@f1a54a32e3d7fd0fd5e1a6704ced4004dd3dcf91) ([merge request](csr/asr4!310))

## 0.6.7 (2023-07-14)

### Patch (4 changes)

- [Revert "now works compiled with torch 2.0.1"](csr/asr4@56495b1799b6bd5c474599920648b16c77fc75b5) ([merge request](csr/asr4!288))
- [[EEE-932] Add testing script for batch long audios](csr/asr4@87a15b7d9919cfad00596f9f7e43fd77af2b7c08) ([merge request](csr/asr4!284))
- [Add batch option to client.py](csr/asr4@a4aca3a60d427aecb2494ce571ac596c632a89d2) ([merge request](csr/asr4!281))
- [Update to models of 10s and expected metrics with these models](csr/asr4@9817a01cbc46e2aacca0be3f76968d530e99d398) ([merge request](csr/asr4!285))

## 0.6.6 (2023-07-12)

### Patch (2 changes)

- [[EEE-894] Add heuristics to timestamps calculation](csr/asr4@6ddf82ea2eba086cc347e68ffd7d1d121a35efc9) ([merge request](csr/asr4!280))
- [[EEE-894] Fix time intervals](csr/asr4@a50706f36d3d74337e835b6464eef89a352fc0db) ([merge request](csr/asr4!279))

## 0.6.5 (2023-06-27)

### Patch (1 change)

- [Update VERSION](csr/asr4@4988011876970556579237a22a80b9d36aec030b) ([merge request](csr/asr4!278))

## 0.6.4 (2023-06-26)

### Patch (2 changes)

- [Update VERSION to 0.6.4](csr/asr4@2ba3a676a48f7bbfb525ae8cce82a60c021c8f5d) ([merge request](csr/asr4!277))
- [upgraded base image](csr/asr4@7d0edfef9802f2bfbbec56f87715468ea29050d5) ([merge request](csr/asr4!276))

## 0.6.3 (2023-06-21)

### Patch (2 changes)

- [Update VERSION to 0.6.3](csr/asr4@b4212e47041b7125cda360b79418f5039f92a14f) ([merge request](csr/asr4!275))
- [Refactor/online decoding functions](csr/asr4@07d384165ce2b06175f629b2170839f68141566c) ([merge request](csr/asr4!268))

## 0.5.2 (2023-05-31)

### Patch (2 changes)

- [Update files to 0.5.2](csr/asr4@91a09446859d6fc4cf993c4daeb7b8f8d54c8342) ([merge request](csr/asr4!259))
- [Adding tests for local decoding and fix accumulated timestamps when partial decoding](csr/asr4@6f3c1e25a072f3f6ff13bd6f2479b29a0bf7bacf) ([merge request](csr/asr4!258))

## 0.5.1 (2023-05-29)

### Patch (3 changes)

- [Update version files](csr/asr4@64f9bca3723a24fb0680cf0a3c05ebb9ac7f0134) ([merge request](csr/asr4!257))
- [Feature/make streaming client](csr/asr4@338e5e1ffd3f372d40efe06f722441c9bb32a94c) ([merge request](csr/asr4!255))
- [wordTimestamps ->  timesteps](csr/asr4@385c5861969d3b72c704a83a61f4d63924be5826) ([merge request](csr/asr4!256))

## 0.4.2 (2023-05-08)

### Patch (5 changes)

- [Increased patch version](csr/asr4@ba904e52f9eaa1dbab83314eaf9dab4543cfa96b) ([merge request](csr/asr4!249))
- [Updated pt_BR accuracy references after weights adjust](csr/asr4@7125fa56987fd6e7fd93075ec38a90b618b60f27) ([merge request](csr/asr4!248))
- [Adding unit an integration tests for ASR4 formatting](csr/asr4@d4694a79cdc3006aafddcb4c774fd81ee7a817d6) ([merge request](csr/asr4!246))
- [Now we can process subwords](csr/asr4@33862df5fcf749932540271b43c30c4163bc5e74) ([merge request](csr/asr4!225))
- [Refactor server flags](csr/asr4@7ed91c879c02e6ac56f9bb09417298c0bf49c1ba) ([merge request](csr/asr4!243))

## 0.4.1 (2023-04-30)

### Patch (2 changes)

- [Fix ES formatter](csr/asr4@86d93aaf9d6ceb49548a42f5ca08a81c5856171b) ([merge request](csr/asr4!239))
- [Update VERSION](csr/asr4@b87b003d9126cf5a822b47d663243b9919216e95) ([merge request](csr/asr4!236))

## 0.4.0 (2023-04-18)

### Minor (1 change)

- [Optional formatting in query](csr/asr4@75f4f1794c06ad36de6b3774479e02d4aa7f2600) ([merge request](csr/asr4!228))

### Patch (4 changes)

- [Lower accuracy metrics for ES because CPU is worse than GPU](csr/asr4@7545fcaca60e8e89cfb4210984f5e3f56cac8a58) ([merge request](csr/asr4!231))
- [Bump version to 0.4.0](csr/asr4@fc44a2b6baab856cbc0f2010adad9e93c2bb7557) ([merge request](csr/asr4!230))
- [fix/ Update E2E ES Upgraded test metrics](csr/asr4@28b802cf86e86b1de065714723ef728d4000955c) ([merge request](csr/asr4!227))
- [feature/ Added ES formatted references GUI to E2E testing data](csr/asr4@5e7485176cece2d9063b84ab4c81328d341d0f11) ([merge request](csr/asr4!226))

## 0.3.1 (2023-04-03)

### Patch (1 change)

- [feature/ Add duration field to the ASR4 Streaming Response](csr/asr4@59b6b04b7629fa066adad939c31070e329e4d03f) ([merge request](csr/asr4!222))

## 0.3.0 (2023-03-31)

### Patch (2 changes)

- [feature/ Add duration field to the ASR4 Streaming Response](csr/asr4@59b6b04b7629fa066adad939c31070e329e4d03f) ([merge request](csr/asr4!222))
- [Bump minor version](csr/asr4@ae4b6d22d957a0a3e98754b46225bab6c597048f) ([merge request](csr/asr4!223))

## 0.2.13 (2023-03-27)

### Patch (1 change)

- [Update expected metrics](csr/asr4@6949d0ac461b2e30f145abaab9fbcff62ebce08a) ([merge request](csr/asr4!221))

## 0.2.12 (2023-03-27)

No changes.

## 0.2.11 (2023-03-20)

### Patch (2 changes)

- [Add LM to EN-US](csr/asr4@de8bb1d3db15459b0eb6839a8a3dcb52966d5d66) ([merge request](csr/asr4!218))
- [Fixes output format issues when using language model](csr/asr4@9f603454ec4fb67744e86b3571183492c891b72f) ([merge request](csr/asr4!217))

## 0.2.10 (2023-03-16)

### Patch (2 changes)

- [fix/ Replace invalid Speech Center ECR Repository Name](csr/asr4@505e0ab906e916752e95cab50d96f3ddf3bb3e4b) ([merge request](csr/asr4!216))
- [New AWS account settings](csr/asr4@292b8448640801abbde6c4dffdce0bf4200a28f0) ([merge request](csr/asr4!214))

## 0.2.9 (2023-03-10)

### Patch (3 changes)

- [Update formatter version to have logs](csr/asr4@3ed1582c978f923943a68151c4dffb70b822fba5) ([merge request](csr/asr4!213))
- [Feature/eee 456 using lm in decoder](csr/asr4@79bd3e2915706740160421f4c6c4a8483a1eb0ed) ([merge request](csr/asr4!203))
- [Feature/eee 456 using lm in decoder](csr/asr4@7f48ade3d98fc26fc5747471882545a6fdf064ac) ([merge request](csr/asr4!203))

## 0.2.8 (2023-03-08)

### Patch (8 changes)

- [upgraded patch version](csr/asr4@6d3f65f5cef10c793ffe6a3079c4cd18d6d912cf) ([merge request](csr/asr4!208))
- [fix/ Auto-Generated Release Notes from Tag Pipeline](csr/asr4@b0f74dbbd10c59e98a27091d9169de3a85f76f4a) ([merge request](csr/asr4!206))
- [Add everlapped chunking and better logging](csr/asr4@6c2dd9b3f8e718ccd8e866217863c091cdac1282) ([merge request](csr/asr4!205))
- [Add everlapped chunking and better logging](csr/asr4@7cb3205bfd85b73dbbfaffee47fd2d3d4226b26c) ([merge request](csr/asr4!205))
- [Update README.md](csr/asr4@298f32a33815c86206c4d92c9e8af57a94d19ef7) ([merge request](csr/asr4!206))
- [Feature/eee 456 using lm in decoder](csr/asr4@79bd3e2915706740160421f4c6c4a8483a1eb0ed) ([merge request](csr/asr4!203))
- [Feature/eee 456 using lm in decoder](csr/asr4@7f48ade3d98fc26fc5747471882545a6fdf064ac) ([merge request](csr/asr4!203))
- [Update .gitlab-ci.yml](csr/asr4@c7b13d890832c84b3ec2be205c5822c322f045f4) ([merge request](csr/asr4!206))

## 0.2.7 (2023-03-01)

### Patch (2 changes)

- [removed unnecessary parallel matrix](csr/asr4@0cd660e4805073c3754f4fb196b804c9d465e9ee) ([merge request](csr/asr4!201))
- [Increase patch version](csr/asr4@353010c934631b7dbeee16be44628bd4cf0e1f40) ([merge request](csr/asr4!202))

## 0.2.6 (2023-02-28)

### Patch (1 change)

- [Feature/cleanup with labels](csr/asr4@36b9eab4d81bec7a35c53bf2a7404a36a1f8332e) ([merge request](csr/asr4!199))

## 0.2.5 (2023-02-27)

### Patch (1 change)

- [added us-east-2 ECR](csr/asr4@069d649d63a1a69d721f152edc45669040acb0f5) ([merge request](csr/asr4!196))

## 0.2.4 (2023-02-27)

### Patch (3 changes)

- [added us-east-2 ECR](csr/asr4@069d649d63a1a69d721f152edc45669040acb0f5) ([merge request](csr/asr4!196))
- [upgrade patch version](csr/asr4@b5a99104ecb8e4eb34870b82461ac75be6456d45) ([merge request](csr/asr4!194))
- [Change amazon ECR address to us-east2](csr/asr4@5fd0f6381a20e3472edbf6a9b8ab3aa9246c7fd8) ([merge request](csr/asr4!193))

## 0.2.3 (2023-02-23)

No changes.

## 0.2.2 (2023-02-22)

No changes.
