# Random chunk samples

10 chunks drawn with seed 42. Review: section number, title, section path, table formatting, and that the text belongs to the section.

## 1. `23501_5.21.2.3_001`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.21.2.3  title: Procedure for AMF Auto-recovery
- section_path: 5 > 5.21 > 5.21.2 > 5.21.2.3
- merged_sections: -
- type: text  part: 1  tokens: 248

```text
[TS 23.501 | Rel-18 | §5.21.2.3 | Procedure for AMF Auto-recovery]
In order to try and handle AMF failure in a graceful manner (i.e. without impacting the UE), AMF can either back up the UE contexts in UDSF, or per GUAMI granularity in other AMFs (serving as backup AMF for the indicated GUAMI).
NOTE 1: Frequency of backup is left to implementation.
For deployments without UDSF, for each GUAMI the backup AMF information (in association to the GUAMI) is configured in the AMF. The AMF sends this information to 5G-AN and other CP NFs during the N2 setup procedure or the first (per NF) interaction with other CP NFs.
In the case that an AMF fails and the 5G-AN/peer CP NFs detect that the AMF has failed, or the 5G-AN/peer CP NFs receives notification from another AMF in the same AMF set that this AMF has failed, following actions are taken:
- The OAM deregister the AMF from NRF indicating due to AMF failure.
- 5G-AN marks this AMF as failed and not consider the AMF for selection until explicitly notified.
```

## 2. `23501_5.4.4.2a_001`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.4.4.2a  title: UE Radio Capability Match Request
- section_path: 5 > 5.4 > 5.4.4 > 5.4.4.2a
- merged_sections: -
- type: text  part: 1  tokens: 178

```text
[TS 23.501 | Rel-18 | §5.4.4.2a | UE Radio Capability Match Request]
If the AMF requires more information on the UE radio capabilities support to be able to set the IMS voice over PS Session Supported Indication (see clause 5.16.3), then the AMF may send a UE Radio Capability Match Request message to the NG-RAN. This procedure is typically used during the Registration Procedure or when AMF has not received the Voice Support Match Indicator (as part of the 5GMM Context).
NOTE: During the Registration Procedure, if the AMF does not already have the UEs radio capabilities and if the RAT where the UE is requires the establishment of AN security context prior to retrieval of radio capabilities, the AMF needs to initiate "Initial Context Setup" procedure as defined in TS 38.413 [34] to provide the 5G-AN with security context, before sending a UE Radio Capability Match Request message.
```

## 3. `24501_5.3.5.2_004`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §5.3.5.2  title: 3GPP access service area restrictions
- section_path: 5 > 5.3 > 5.3.5 > 5.3.5.2
- merged_sections: -
- type: text  part: 4  tokens: 414

```text
[TS 24.501 | Rel-18 | §5.3.5.2 | 3GPP access service area restrictions]
of the user-plane resources via 3GPP access is allowed; for all other cases, the UE shall indicate for each PDU session in the Allowed PDU session status IE that re-establishment of the user-plane resources via 3GPP access is not allowed; and
2) if the UE is in 5GMM-CONNECTED mode or 5GMM-CONNECTED mode with RRC inactive indication over 3GPP access, the UE:
i) shall not perform the registration procedure for mobility and periodic registration update with Uplink data status IE except for emergency services or for high priority access;
ii) shall not initiate a service request procedure except for:
- emergency services;
- emergency services fallback;
- high priority access;
- responding to paging or responding to a notification.
The UE shall not include the Uplink data status IE in the SERVICE REQUEST message or CONTROL PLANE SERVICE REQUEST message except for emergency services or for high priority access. In case of emergency services, the UE shall indicate that uplink data is pending only for the PDU session for emergency services;
iii) shall not initiate a 5GSM procedure except for:
- emergency services;
- high priority access; or
- indicating a change of 3GPP PS data off UE status;
iv) shall not perform the NAS transport procedure except for the sending:
- 5GSM messages for UE-requested 5GSM procedures allowed in iii) above;
- 5GSM messages for network-requested 5GSM procedures;
- SMS;
- an LPP message;
- a UPP-CMI container;
- an SLPP message;
- a location services message;
- an SOR transparent container;
- a UE policy container;
- a UE parameters update transparent container; or
- a CIoT user data container; and
NOTE 1: The contents of CIoT user data container can be data that is not for exception reports, or data that is for exception reports if allowed for the UE (see clause 6.2.13).
```

## 4. `24501_4.6.2.4_004`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §4.6.2.4  title: Network slice-specific authentication and authorization
- section_path: 4 > 4.6 > 4.6.2 > 4.6.2.4
- merged_sections: -
- type: text  part: 4  tokens: 233

```text
[TS 24.501 | Rel-18 | §4.6.2.4 | Network slice-specific authentication and authorization]
using the generic UE configuration update procedure as specified in the clause 5.4.4 and inform the SMF to release all PDU sessions associated with the S-NSSAI for which network slice-specific re-authentication and re-authorization fails or network slice-specific authorization is revoked;
b) if network slice-specific re-authentication and re-authorization fails or network slice-specific authorization is revoked for all S-NSSAIs in the allowed NSSAI or the partially allowed NSSAI but there are one or more default S-NSSAIs which are not subject to network slice-specific authentication and authorization or for which the network slice-specific authentication and authorization has been successfully performed, the AMF updates the allowed NSSAI or the partially allowed NSSAI containing these default S-NSSAIs and the rejected NSSAI accordingly using the generic UE configuration update procedure as specified in the clause 5.4.4. The AMF shall also inform the SMF to release all PDU sessions associated with the S-NSSAI for which network slice-specific re-authentication and re-authorization fails or network slice-specific authorization is revoked; or
```

## 5. `24501_3.1_011`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §3.1  title: Definitions
- section_path: 3 > 3.1
- merged_sections: -
- type: text  part: 11  tokens: 447

```text
[TS 24.501 | Rel-18 | §3.1 | Definitions]
MBSR-UE: a UE that is operating as a MBSR and supporting UE NAS functionalities specified in this specification.
Mobility registration for disaster roaming services: A registration performed with 5GS registration type "disaster roaming mobility registration updating" in the REGISTRATION REQUEST message.
MUSIM UE: A UE with multiple valid USIMs, capable of initiating and maintaining simultaneous separate registration states over 3GPP access with PLMN(s) using identities and credentials associated with those USIMs and supporting one or more of the N1 NAS signalling connection release, the paging indication for voice services, the reject paging request, the paging restriction and the paging timing collision control (see 3GPP TS 23.501 [8]).
N1 mode: A mode of a UE allowing access to the 5G core network via the 5G access network.
Native 5G-GUTI: A 5G-GUTI previously allocated by an AMF.
Non 5G capable over WLAN (N5CW) device: A device that is not capable to operate as a UE supporting NAS signalling with the 5GCN over a WLAN access network. However, this device may be capable to operate as a UE supporting NAS signalling with 5GCN using the N1 reference point as specified in this specification over 3GPP access. An N5CW device may be allowed to access the 5GCN via trusted WLAN access network (TWAN) that supports a trusted WLAN interworking function (TWIF) as specified in 3GPP TS 24.502 [18].
Non-CAG Cell: An NR cell which does not broadcast any Closed Access Group identity or an E-UTRA cell connected to 5GCN.
Non-equivalent PLMN: A PLMN which is not an equivalent PLMN.
Non-equivalent SNPN: An SNPN which is not an equivalent SNPN.
Non-globally-unique SNPN identity: An SNPN identity with an NID whose assignment mode is set to 1 (see 3GPP TS 23.003 [4]).
```

## 6. `23501_5.30.3.5_001`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.30.3.5  title: Support of emergency services in CAG cells
- section_path: 5 > 5.30 > 5.30.3 > 5.30.3.5
- merged_sections: -
- type: text  part: 1  tokens: 273

```text
[TS 23.501 | Rel-18 | §5.30.3.5 | Support of emergency services in CAG cells]
Emergency Services are supported in CAG cells, for UEs supporting CAG, whether normally registered or emergency registered as described in clause 5.16.4 and in clause 4.13.4 of TS 23.502 [3].
A UE may camp on an acceptable CAG cell in limited service state as specified in TS 23.122 [17] and TS 38.304 [50], based on operator policy defined in TS 38.300 [27].
NOTE: Support for Emergency services requires each cell with a Cell Identity associated with PLMNs or PNI-NPNs to only be connected to AMFs that supports emergency services.
The UE shall select a PLMN (of a CAG cell or non-CAG cell), as described in TS 23.122 [17] and TS 23.167 [18], when initiating emergency services from limited service state.
During handover to a CAG cell, if the UE is not authorized to access the target CAG cell as described in clause 5.30.3.4 and has emergency services, the target NG-RAN node only accepts the emergency PDU Session and the target AMF releases the non-emergency PDU Sessions that were not accepted by the NG-RAN node. Upon completion of handover the UE behave as emergency registered.
```

## 7. `23501_5.17.7.2.3_001`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.17.7.2.3  title: Relaying
- section_path: 5 > 5.17 > 5.17.7 > 5.17.7.2 > 5.17.7.2.3
- merged_sections: -
- type: text  part: 1  tokens: 66

```text
[TS 23.501 | Rel-18 | §5.17.7.2.3 | Relaying]
The AMF performs relaying between N2 and N26 messages as described in TS 38.413 [34] and TS 29.274 [101].
The MME performs relaying between S1 and N26 message as described in TS 36.413 [100] and TS 29.274 [101].
```

## 8. `24501_9.11.3.31_003`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §9.11.3.31  title: MICO indication
- section_path: 9 > 9.11 > 9.11.3 > 9.11.3.31
- merged_sections: -
- type: table  part: 3  tokens: 236

```text
[TS 24.501 | Rel-18 | §9.11.3.31 | MICO indication]
Table 9.11.3.31.1: MICO indication
| Registration Area Allocation Indication (RAAI) (octet 1, bit 1) |  |  |
|---|---|---|
| In the network to UE direction: |  |  |
| Bit |  |  |
| 1 |  |  |
| 0 | all PLMN registration area not allocated |  |
| 1 | all PLMN registration area allocated |  |
| In the UE to network direction bit 1 is spare. The UE shall set this bit to zero. |  |  |
| Strictly Periodic Registration Timer Indication (SPRTI) (octet 1, bit 2) |  |  |
| In the network to UE and the UE to network direction: |  |  |
| Bit |  |  |
| 2 |  |  |
| 0 | strictly periodic registration timer not supported |  |
| 1 | strictly periodic registration timer supported |  |
| Bits 3 and 4 are spare and shall be coded as zero.<br>NOTE: In the network to UE direction in the CONFIGURATION UPDATE COMMAND message, bits 1 and 2 shall be coded as zero. |  |  |
```

## 9. `23501_5.15.17_005`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.15.17  title: Partial Network Slice support in a Registration Area
- section_path: 5 > 5.15 > 5.15.17
- merged_sections: -
- type: text  part: 5  tokens: 337

```text
[TS 23.501 | Rel-18 | §5.15.17 | Partial Network Slice support in a Registration Area]
- The UE is considered registered with an S-NSSAI of the Partially Allowed NSSAI in the whole Registration Area. The UE does not trigger registration when moving between the TAs of support and non-support for the S-NSSAI within the RA.
- The UE is allowed to initiate PDU Session establishment for the S-NSSAI only when the UE is in a TA where the S-NSSAI is supported. If the UE has overlapping areas between non-allowed area, area where the S-NSSAI is supported, then the non-allowed area restriction applies.
- If the AMF determines a PDU Session is associated with S-NSSAI present in the Partially Allowed NSSAI, the AMF indicates to the SMF that the PDU Session is subject to area restrictions for the S-NSSAI. As a result, the SMF subscribes to "UE mobility event notification" event for reporting UE presence in Area of Interest by providing S-NSSAI to the AMF as described in clauses 5.6.11 and 5.3.4.4 if this event has not been subscribed before. The supporting AMF shall provide this indication in all the subsequent PDU Session update message to the SMF as long as the S-NSSAI of the PDU Session is subject to area restriction. When the AMF does not indicate to the SMF that the PDU Session is subject to area restriction for the S-NSSAI, the SMF may unsubscribe "UE mobility event notification" event in the AMF if this event has been subscribed before.
```

## 10. `24501_9.11.4.8_007`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §9.11.4.8  title: Mapped EPS bearer contexts
- section_path: 9 > 9.11 > 9.11.4 > 9.11.4.8
- merged_sections: -
- type: text  part: 7  tokens: 205

```text
[TS 24.501 | Rel-18 | §9.11.4.8 | Mapped EPS bearer contexts]
Table 9.11.4.8.1: Mapped EPS bearer contexts information element (continued)
When the parameter identifier indicates mapped extended EPS QoS parameters, the length and parameter contents field are coded as specified in clause 9.9.4.30 of 3GPP TS 24.301 [15].
When the parameter identifier indicates traffic flow template, the length and parameter contents field are coded from octet 2 as shown figure 10.5.144 and table 10.5.162 of 3GPP TS 24.008 [12].
When the parameter identifier indicates APN-AMBR, the length and parameter contents field are coded as specified in clause 9.9.4.2 of 3GPP TS 24.301 [15].
When the parameter identifier indicates Extended APN-AMBR, the length and parameter contents field are coded as specified in clause 9.9.4.29 of 3GPP TS 24.301 [15].
```
