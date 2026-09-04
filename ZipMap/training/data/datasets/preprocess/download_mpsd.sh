#!/bin/bash

# Download the public MPSD dataset
DLCMD="curl --progress-bar -L -o"
DESTINATION=$1


URLS=(
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_YbpFOEEq-i0gsdv_rs3pS-YCuxroa4SjldwIg6Afpsk5OvkKW1qik7FV_VUB7d5MpMoCsToCqmAp92MazElVn0F_d_Sx06yqB8R1hckzF4A2CEiZMLd76ikNVdSzJQiUolJ8l.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdkHrDhby3iRYal_2b6if_DYRM0cIQPEZVV2CDQZLgIrwWSNYoCgxJg3iNNwCWtDD0otONdliOgw3U-D14N0ngvE&ccb=10-5&oh=00_Afa2Zeka_Q1tlJDwbYsmAOQr7Zb3qAOWxX7wOjFXGiHg6g&oe=68F669C7&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_HR43KZvA3xlrihi_Uwy0S1kGfDkPg_4n1nQn8zrJRJGFoQqzUyeBbz8JXIpT6NMNwaJ57rpMO1CZFspPVTntgn_inIIXvstEFjl6EAd8J9VxmaBcsNvGxYXDHAERKwg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdmfyWFuvCtbxHKt1Bk3vdR14s_k0NlgOyEgtRojjlgKK4cuqjBjarxSko-x0Cp99nwLt2WGt9WcpQWJ5lg41aG2&ccb=10-5&oh=00_AfYKn9N5OszYU5MGt5ZnoH2pB5go9TJuDD9Iu9jhkztXQQ&oe=68F6781A&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An8f5aynDuONwsdgQ8fjK63SNyklj2f1FKoL6Vo35wj_S4dE9vnp6dT1Kr3JS7eXrjxS92WG4NJqobL2pPP0XnKnxk4cfriZ5bZZgCSt3jKeypQixBR1qDvZ71v87Q7Z2g.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdnasTeCQB8oYfALtkRtShwc4ZTMdxfd20mYUmN_HbnPsE_U7jwpRaPc3VM8II9SRmwLPpdY7rk80W9k5ot9OCfH&ccb=10-5&oh=00_Afa_NzvJf1NzetfsLlexsKXHLO5_gPnffUzwPUoD1XnpOA&oe=68F672F5&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An8NneZmJ_hjuskNMEv25epzZ_b6mQW-aJlATjC7p6KtwzOQ3IkVFFwXIobRRgKWLUYw8D43nRnHxcjD9c434smf2KgqZrPcuMec0Mmejc_xEmB899tt_1EvmYq8rqY4tA.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adm4mEOe2e-HizbnYVLSZ5L9MoQ5YR0lWH5JDKfVu3SuL1Se9_A28-c5sNjCQfgxhkGG9iLw22NXQoZWoM74q-FQ&ccb=10-5&oh=00_AfbcgBfY1xl69XpLwmQPsqm5lFTToMr1uXgfjziARA3SLQ&oe=68F68E66&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An8TgTgFecJ2AZWinsQiSSGc2mULvYCFuwN8ExG415Uu5lSuk2Gw8-ftcOTfRcCmXskzxgvAIC4TQad3iMpT3XQQejOlOV0XazrzIrjhVV0HFLVcirYH0VS5ToeG1ndG8g.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdknPthrk4RawGOw4rvfYuULoernOXne5kO58zsJyMUWgOQ8rWM_pEeb4GcBUxUapUZc1E3ef6_2PxBFi6-zSj5O&ccb=10-5&oh=00_AfaQotET2PMsRFnPM3fIcEqKFW0N5I1PNcmgMu0YPCd9MQ&oe=68F67A64&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An8nbpVa2qxWwflhy6Z4jm_zdHya4_O_dB_fjaQTf-7AM3tZpz6Kwxb9Y5s3t6KX_CG4YxUtn6Wg4OyoGc22O43Thtzm4GWdfx_jVE2Zx2nJkaTn51c5JanGdqfv3XmQhg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adnmo8zJ8X0T0JiM1qCHqDMlaXjcweEegoGMznOUdwSZ9-cNJgT0ATTdCU2KJ1q2Z8F-D_sbbxEDDl-aZM1EUfct&ccb=10-5&oh=00_AfYFTbGzDnk3zHuoJAot36fuVinvhRt6Y5JNSqJzPMjSXA&oe=68F679EF&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An-zKIdtRtoB361CWl7fbktbdst8zAvszgbnYJloYkK9OQp2XhTYjr3kJUuQDjptjDvAiR-Juq7ty5a6H_eFCUT9pPRqWs833V9mOvwBvabQkIWvdMFPRfMQiV2oxwNPdA.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdlYphRvi2Wl3EVnVCQ0YTuHCSsM1DuO3gkESGJyed0vrhg3OTwFEeGmuOuBMAh9A48mKjSJOiTN34ufWJLXxYKf&ccb=10-5&oh=00_AfaZddGSeGTD4NJm9EiUqqORsQDKZeJhXvL4t4O5A7cFzw&oe=68F6647F&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An9z6qlJnGRkaSqZM0KtWeGI9Shmw-paQJl_tOj6WgsWIoweUrohszuFDCXFYjwRqRfx1RnEBQrSQ7_IP5A6FI4nPEMxpoO_2H6Kyguccn4TM_r3uzZ7vI1zvAxM3BBQxQ.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdkXbfiKOFIOSoUTyuNMx46ySGyXZXBGXvryaxcfzWUqE2keCjGDnRl_UkOQSDf3GQYhQq_4zB_joso9_u-V0B-0&ccb=10-5&oh=00_AfZOUzJiMNgWO_Kp77uOqkUux0WfklatIlEzyDxbKKA7bw&oe=68F66912&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_bMd_9QtE6ESdH51XUHAZzuwDQSgOmYLdFWSMhGEX62AlISSoe588--boiK187zKsJiMrDgL1lYsfQG7dKlLG8nHT7X5hSEypBmuzlMavJDw3Yzn923hJwINeNOuu1Lg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Admo5dNYjWQ8DINPqzUL3FdbK3pMPbOmZAKUH-evj8MtWvuxL_HKOV3bamhlHi72WGwtCTYzaee_3bK-U-qCfeOT&ccb=10-5&oh=00_AfYu2jmJkfg57fLwTIaBll4TE8xcq2WtRXRPvkfQjEEdHQ&oe=68F67B0D&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An87OsgbLx1KPfZzH5jtKbK-CQVwpaUNDtQdPGEi5hteVcVu5GpBEVYv6lDpf8Xof1nH782HJc6Eq5JLOt_9RtOYt1pE4mD6nLpPrGXtQZE92ixlhrpBLLjwlLFYBidjfg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adk4JONcK57ln13LvRQ-qRGoIVQTPUMzR43TwoQGxTi6Pcyiv3NWE69B0e3FrqdlaN60Vi-rvqAyH5WNZ6CAilzv&ccb=10-5&oh=00_AfYG-XtPEQGKgujarUMSsf9apcZVXtGfoLkQ6fJAaeZVIQ&oe=68F68C12&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An9PSHwIOpAEgCGYvTPxWqme7Y_4HESPDA6gFN5rRW1aO1ptY_GCfsKB3z5CvdLTnADxwNnU-eyqdk2POcE7pe677jhKYhQio4z7Df9Hgl7aCE2W5LWWbASzclJQhkIh-w.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdmTG4NSqXb74OQHyrUM-cypXYXDWDo5M_yHUVfgnEnDjAfic2Y3ZgzNs9pfiDiKJVTUDtY-jSHmk2rkLSiFNtI5&ccb=10-5&oh=00_AfZe1I5JYXKFBHQgN5C00BRgUI3-axpB67Smi_jSOfco9A&oe=68F662B9&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An-DEZAEuCJ5jYES0zfV-pv587dBOWu2Z0xUoQcB3_oHjtwKruwaU7Xa57xkyMNIyg3Hxw1cJCG4gBlsTTQOQ_WBtKgl-VxDP1H8NPUDiNFdsrjc3SrDnO0j81k9a9Ak9A.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adna2gvM6BnNpEfKtKy4-36PbSn5AMrQGbaaOTRpf5xWi4a2HcoWa2fqtIVNQ2um6Sfw7PCW4n3KWbHkOvg-r_M8&ccb=10-5&oh=00_AfZtuctCjxzVGGPdtn4Lu1q2E-DwNv_T-m4dx0c8Hr1lJg&oe=68F67A51&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_NJRYjkXlnBDptGen4KOdJEWFnrv5evNufx5JkcIo4fBBfTLIlZQuReM9Ongjp8UnDfXcc9yN1qu3ccFgKx09Pc8VzRxSs77NtmqmPVP79bni9JXRwyc-g27qfUrPDhg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdnDvBylKfzyPe4JibaL4buu1MeEL_Gy0m6RAdhUrV7yQcDDlH4of1tvFZ32vovmKn2JskaEOF9SI2mK-kYQBN6_&ccb=10-5&oh=00_AfanuqhJUD6zyiU0HWgsehyESkTCIpNhgujks80EskN4hA&oe=68F65FF5&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An8EmMi5r-n_AdDUkudogHS-dF5tEp94IVcsRq1VIJ7YC76NE9CwuY1s8bS2tYgUfx-NEJA1glHVpJCfeUGz286vIYhh3KU5ajdpH1D3yrhI0plA-4KXY6Qx_1HJQjfErg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdmOB8sEqQmN9LcYv7-iyD1-hQp3GNLKzCCCbXxJf7vsEtuSqovI2tCZJdBzEkqy6N_0tVXFfgcH-xHUdGXpmbe9&ccb=10-5&oh=00_AfblF6MBsuBnf5cuPo9azdN_ajEA8EUqytzTBPNeoJxlXw&oe=68F66DF1&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An9Z9KiE_joAu5lTP0GQ2iYPdjip4jhXQW06usP9iat_8OtH65eQzz8DOHXdYZsT6tgFUh_bLXUledmecEWC4lBJI25ntsWeHMf0sKst6qhw-_CSzjn2C2osCi6aCL_XMg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdmqBWYBfbIkZH_yzWds7bUFarmwlbMOS0DQw_Fcmp8feQzyhdGPayc4scK3YXvKBJie2B3jUcyMlpue94mE6Atb&ccb=10-5&oh=00_AfZmRV2cEF3DncUzi9w_sQxX-l2Df73Fo6rDCqtyGAHwQw&oe=68F6807C&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_8A6uBpoK60mPc2hoHnR--jSDa7kfiMVhP3oUuP6XCbS8iaVz07qm6fAvuFYpmGwaBVjXlYzwOgDDBSjoSYd7ovkUot0vVYmVby3n_AgRm4G4tnAWZbDCRH97b-Um1Sg.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adk-sNw2KfRwLNFTsubL0_pi09ez5fqTp6gKTu72RSZc4BZVxorSQGlR3cQNec8a0S20rIGQ9F7iEdxJ2lGGI4RS&ccb=10-5&oh=00_AfbHx-_l5YX0rESS_hAwyWrnUS45OU0hesnOJ1bp8H05hg&oe=68F665C4&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An-f-Omiq9zbgv5KoFC-k2A7SkE0WKISDTUwaV36VRyOaYhkH94dqwf2GSpjEXsmE2zMbjX3Tif5NJsWl3IcrNGui6kFzwMqZP_8LH1ChWIDgFo9W5i2wi_m8Ssv9v5D2A.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdkXyv7R3AsClzBpHsdonIAQhhYWoZFN8PwwjRhkRdGTwLdukohxOJZkmLBCsZ7-aYR3xEZWHpEzzf_WuysxUcKW&ccb=10-5&oh=00_AfZ9ZzzFCu1opqyhr9ECYf_-g82RGT6mPQm5iXFpMYqExg&oe=68F681C4&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_dYDkJhKg5xO5VsMO2LLx2hROi2BSSRg5qVxofmkzZiY8yzx2ozkQrTAHWoeH61fvlGcvWHnXTaZIlYvhvNrnI_6FBP7xHyHR5oX4PJdHTlNK0bKyG4TZlr987uC1tyw.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdmUdBU2stxrVe9B4TvPgKlUG4YLcH6Z1IM3snt8c0b8jhqFivx932H9RutHFitP_mEBVwH86wx3pS2DUVFYWVw9&ccb=10-5&oh=00_AfZuCg_4pgrzjf9e2dYfmUTVqu0QfPvFzOi8mZA_gYjXyg&oe=68F6820A&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An-gdsEZGIKTWXktf_8lEyjb-3eBxYFikg1ItQH-2DVdgeRYSl0YStSN5c8B0KVNwW-8_RHNzhB4-aqmPZ8lrW1-n0H0LcRVFMFLRxgpBk4zqVun1YcnLms_U199v1hJ_g.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adl6crHDfHiP3jvEoAAvou9Ju6hogwEp0PZzpzozegaL8Adnd5V0MU2yfYKwmFdUyk33uKfteOBPZ2t9I8SuT9M1&ccb=10-5&oh=00_AfZqIBuEu1_BQfCd_AG6BIi07IzKY0gx6q_puN2DDGvBXA&oe=68F6737F&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An8J2lu0p-zprNwwWSOkKHOodiMwD1dM72EUDogd84laMaTaL4cqBenmiTmHGO8yhPqfN-dsjC-eokUz0cuW9sgly0K6HpYRO3qNlhTlExo6ovUF-MolC6dXFzsIJt8_WA.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adl5qqDulLL6rKnNYppidrBnrv73c3OV_7zh_21AgkaPBBJQp1-CnVD0kFoT2owMMbpEODp8w19-UBtEcOUX_4qC&ccb=10-5&oh=00_Afb62Q55b4rfmRLGlQjmf60qJ-jDiLW7EOUrog8_obs6tQ&oe=68F68DC4&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_-eapSSSigmRDBycBHrMBsN6LB2pRqgxurYgpAaoHZXx-n52_QjkJ-Z7ygt-Zn8Ek-EpnWbM1tMYSa-nsVGsmFqrL8dgnmz0iCaUSxJs5f04Sx_C0gJC0ZeNwEQLLEjA.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adn0suxAIIYbJbVMHpwwyotcgRZfqrNZfGIJ2A6rXqvDHtE9fE4NENQG7jZq68-I55wsWXJP4LVpThiM_mlrApjl&ccb=10-5&oh=00_Afb1_zIhn-2lo5_ydyENVcqCnyz0b2n45iHLK5thRYdvIQ&oe=68F662B7&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An-FPUv69q0Em3Y4Hhbl9e__EWslRgHCFBAmGvvkeJ-lKg5tPOLxtSKnRwJMiPZdtQPJVurq2gJ-L3gkrb4CeKDefi_bFmKK3NQ-TCAonnl2OXFvDTd_6gRUOOXUKETvaQ.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adn9PcM-iK9E13x_rP_zWB2AuiGAXMr8t0WlH6r8cvzm0bhetz1MVCwePMewHeJhvTAsedZ5g6YNuQpSjPT6bN-s&ccb=10-5&oh=00_AfbMET-My03jkazoEtNBFrTF3_15IwCyZuc-gH8vlw8dDA&oe=68F69196&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_zGVbA1qIFWZIUj-yA3Vh76AW41pEn2KH2aIQFMwp9HzVd20qIdvlOeL5cLvDxmx4QeUa-W4Z0xVXkic5nuq_aKLF1zVLNXRH3Esg_RxSiMGxTXIc53hSi2_sbogUWRw.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdmFO0gGMQsPJNNejtUhWJpDjuyIf4LtOv8lW88RWYf2WuSfW0vrvL0cK4ZZZs7uMWhEMtzkKcjiMVnikbNNtbsP&ccb=10-5&oh=00_AfYCsnHC5HCNZt9RbspyGjkIjCtob0-zZLlZST3GMk2uKA&oe=68F67A1C&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_OA25H1KBJ62HzkrTZ4srXneUiV5SFpmNFvrSHLgn_TK4TGdyA13g4XTh-XQ3lTbCVsCQyAnnhNzP7g4T5rrAJ27vrR-lgxvQDiMOdulC4zZaxKIKxlzc.md?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdkfLtOLf_MjEOQXKob15fUEKeHb-zwFKieq2VdDHg_htJveawqKhz2CfTvaZpz0mrE-q_ZoWlAWBmoRpzSlwmq-&ccb=10-5&oh=00_AfY-dBqUY1jaPSEXkWVa9M9SVlmzoxgJNFhhMdoM4BR39w&oe=68F67F4C&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An_dVt3rUNN8wJdzvvNcwfgQh-ebUW0RkRjNZ6ZSgbi96we3LySO_i7h3OtpFzKeyF9kNOyE3utbLt2NbsG-7x4eHVi53XmSKzZxbxFBtslxquShe2HuLFNOV4tB.zip?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adk_DLIF6H7yT2wKPXOapoRFBmXZ54mi56AYmGTy4X6-wLHziF1lxCrPA9AQJeKGJ3_5S9pBKrNaqOMKIEavC8h1&ccb=10-5&oh=00_AfYUz2seBKDHmxoWVDwZuEsriBIES72_TP153TU9ILQaAQ&oe=68F67AA7&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An84DdKV175yfWDUytmStxkGTz6ngMGh5Hl8aIFJImwp6XW4_Hvjwfpbzr0qrhmyllGNBAxqez2-9iuMypYLSxDrhRi781eEXx6nCuGDKe4ml5oWhzHOZspWCZdfxr08Z21c.npy?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=AdlPE0UsBqLXQvQ0B4NeGABMcjJzZrXuONLpwgw2phsKQQUSeS01TcJ-YrlJuURVmtCUBVwRHJzL_metWC2IOQlE&ccb=10-5&oh=00_AfaQYAkrh2lNtcfVU_uYaGfrtYXc3Tbe_Y58l0VMx1D7Sg&oe=68F67FEA&_nc_sid=6de079"
  "https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An9ruFSo1yDOKXxj1WxfVc8jjyQ17umUADOf2BOKu0r04OCOJdfpSpUtk99BkPoywXT05JD2fCjutvepif7j88s0NQv52xZzcBPto-Zi5XVdi0RzP-z5vPR7Yq00Yo16.npy?_nc_gid=m12OyDzqJVRYOX3m4MDSRg&_nc_oc=Adm1tHwSm4aIn4XMrhBVPk6os3HX13iKTUSxPq6ilfqrGE0TGNMIlTC9EaSw-V6o4KIlWIUvmBUGLYwIhMQA3enI&ccb=10-5&oh=00_Afae7eN3Z8470JyGvTBNiFqcC8XUBa6ICBtSr07wPA5njA&oe=68F6895F&_nc_sid=6de079"
)

# Check if destination parameter was provided
if [ -z "$DESTINATION" ]; then
  echo "No destination directory specified. Usage: $0 <destination_directory>"
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DESTINATION"
if [ $? -ne 0 ]; then
  echo "Failed to create destination directory: $DESTINATION"
  exit 1
fi

for URL in "${URLS[@]}"; do
  FILE="$DESTINATION/$(basename "$URL" | cut -d '?' -f 1)"
  $DLCMD "$FILE" "$URL"
  if [ $? -eq 0 ]; then
    echo "Download of $(basename "$FILE") completed successfully"
  else
    echo "Download of $(basename "$FILE") failed or was incomplete."
  fi
done

echo "All files have been downloaded to $DESTINATION."