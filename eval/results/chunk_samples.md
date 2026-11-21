# Random chunk samples

10 chunks drawn with seed 42. Review: section number, title, section path, table formatting, and that the text belongs to the section.

## 1. `23501_5.27.1.3_002`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.27.1.3  title: Support for multiple (g)PTP domains
- section_path: 5 > 5.27 > 5.27.1 > 5.27.1.3
- merged_sections: -
- type: text  part: 2  tokens: 181

```text
[TS 23.501 | Rel-18 | §5.27.1.3 | Support for multiple (g)PTP domains]
The process described in clause 5.27.1.2.2 is thus repeated for each (g)PTP domain between a DS-TT and the NW-TT it is connected to.
NOTE 2: If all (g)PTP domains can be made synchronous and the synchronization can be provided by the 5G internal system clock, the NW-TT or DS-TT(s) generates the (g)PTP event messages of all domains using 5G clock as described in clause 5.27.1.7.
NOTE 3: This Release of the specification supports multiple gPTP domains as defined in IEEE Std 802.1AS [104]. If a 5GS TSN bridge supports stream gates and/or transmission gates as defined in IEEE Std 802.1Q [98], then they operate based on a single given gPTP domain.
```

## 2. `23501_5.4.4a_001`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.4.4a  title: UE MM Core Network Capability handling
- section_path: 5 > 5.4 > 5.4.4a
- merged_sections: -
- type: text  part: 1  tokens: 448

```text
[TS 23.501 | Rel-18 | §5.4.4a | UE MM Core Network Capability handling]
The UE MM Core Network Capability is split into the S1 UE network capability (mostly for E-UTRAN access related core network parameters) and the UE 5GMM Core Network Capability (mostly to include other UE capabilities related to 5GCN or interworking with EPS) as defined in TS 24.501 [47] and contains non radio-related capabilities, e.g. the NAS security algorithms, etc. The S1 UE network capability is transferred between all CN nodes at AMF to AMF, AMF to MME, MME to MME and MME to AMF changes. The UE 5GMM Core Network Capability is transferred only at AMF to AMF changes.
In order to ensure that the UE MM Core Network Capability information stored in the AMF is up to date (e.g. to handle the situation when the USIM is moved into a different device while out of coverage and the old device did not send the Detach message; and the cases of inter-RAT Registration Area Update), the UE shall send the UE MM Core Network Capability information to the AMF during the Initial Registration and Mobility Registration Update procedure within the NAS message.
The AMF shall store always the latest UE MM Core Network Capability received from the UE. Any UE MM Core Network Capability that an AMF receives from an old AMF/MME is replaced when the UE provides the UE MM Core Network Capability with Registration signalling.
If the UE's UE MM Core Network Capability information changes (in either CM-CONNECTED or in CM-IDLE state), the UE shall perform a Mobility Registration Update procedure when it next returns to NG-RAN coverage. See clause 4.2.2 of TS 23.502 [3].
The UE shall indicate in the UE 5GMM Core Network Capability if the UE supports:
- Attach in EPC with Request type "Handover" in PDN CONNECTIVITY Request message (clause 5.3.2.1 of TS 23.401 [26]).
- EPC NAS.
- SMS over NAS.
- LCS.
- 5G SRVCC from NG-RAN to UTRAN, as specified in TS 23.216 [88].
```

## 3. `24501_5.4.1.2.2.11_001`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §5.4.1.2.2.11  title: UE handling EAP-failure message
- section_path: 5 > 5.4 > 5.4.1 > 5.4.1.2 > 5.4.1.2.2 > 5.4.1.2.2.11
- merged_sections: -
- type: text  part: 1  tokens: 408

```text
[TS 24.501 | Rel-18 | §5.4.1.2.2.11 | UE handling EAP-failure message]
Upon receiving an EAP-failure message, the UE shall delete the partial native 5G NAS security context and shall delete the new KAUSF and the new KSEAF, if any were created as described in clause 5.4.1.2.2.3.
The UE shall consider the procedure complete.
If the EAP-failure message is received in an AUTHENTICATION REJECT message:
1) if the AUTHENTICATION REJECT message has been successfully integrity checked by the NAS:
- the UE shall set the update status to 5U3 ROAMING NOT ALLOWED, delete the stored 5G-GUTI, TAI list, last visited registered TAI and ngKSI;
In case of PLMN, the USIM shall be considered invalid until switching off the UE or the UICC containing the USIM is removed.
In case of SNPN, if the UE is neither registered for onboarding services in SNPN nor performing initial registration for onboarding services in SNPN and the UE does not support access to an SNPN using credentials from a credentials holder and does not support equivalent SNPNs, the selected entry of the "list of subscriber data" with the SNPN identity of the current SNPN shall be considered invalid until the UE is switched off or the entry is updated. Additionally, the UE shall consider the USIM as invalid for the current SNPN until switching off or the UICC containing the USIM is removed.
In case of SNPN, if the UE is neither registered for onboarding services in SNPN nor performing initial registration for onboarding services in SNPN and the UE supports access to an SNPN using credentials from a credentials holder, equivalent SNPNs, or both, the UE shall consider the selected entry of the "list of subscriber data" as invalid until the UE is switched off or the entry is updated. Additionally, the UE shall consider the USIM as invalid for the entry until switching off or the UICC containing the USIM is removed.
```

## 4. `24501_4.22.3_001`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §4.22.3  title: Authorization of C2 communication
- section_path: 4 > 4.22 > 4.22.3
- merged_sections: -
- type: text  part: 1  tokens: 422

```text
[TS 24.501 | Rel-18 | §4.22.3 | Authorization of C2 communication]
The 5GS supports USS authorization of C2 communication for pairing of UAV and UAV-C. The pairing of UAV and UAV-C needs to be authorized by USS successfully before the user plane connectivity for C2 communication (over Uu or over NR-PC5) is enabled. For C2 authorization procedure, the UE supporting UAS services provides to the network with CAA-level UAV ID.
The USS authorization of UAV flight can also be performed during the C2 authorization procedure. The UE supporting UAS services provides the UAV flight authorization information to the network if provided by upper layers.
NOTE 1: The C2 authorization payload in the service-level-AA payload can include one, some or all of the pairing information for C2 communication, an indication of the request for direct C2 communication, pairing information for direct C2 communication, and the UAV flight authorization information (see clauses 6.4.1.2 and 6.4.2.2).
The UE supporting UAS services can establish a PDU session for the C2 communication by providing the CAA-level UAV ID and the C2 authorization payload. The SMF upon reception of the UE's request for the PDU session establishment, determines that authorization is required based on the DNN and S-NSSAI combination of the PDU session is for aerial services according to user's subscription data and the CAA-level UAV ID included in the request.
If a UE supporting UAS services uses a common PDU session for both USS communication and C2 communication, the C2 communication can be authorized using UUAA-SM procedure during the PDU session establishment procedure or during the PDU session modification procedure. If the pairing of UAV and UAV-C is revoked, the network shall disable C2 communication for the PDU session. The SMF upon reception of the UE's request for the PDU session establishment, determines that authorization is required based on the DNN and S-NSSAI combination of the PDU session is for aerial services according to user's subscription data and the CAA-level UAV ID included in the request.
```

## 5. `24501_4.5.2A_011`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §4.5.2A  title: Determination of the access identities and access category associated with a request for access for UEs operating in SNPN access operation mode over 3GPP access
- section_path: 4 > 4.5 > 4.5.2A
- merged_sections: -
- type: table  part: 11  tokens: 663

```text
[TS 24.501 | Rel-18 | §4.5.2A | Determination of the access identities and access category associated with a request for access for UEs operating in SNPN access operation mode over 3GPP access]
Table 4.5.2A.2: Mapping table for access categories
| Rule # | Type of access attempt | Requirements to be met | Access Category |
|---|---|---|---|
| NOTE 1: Void<br>NOTE 2: Access for the purpose of NAS signalling connection recovery during an ongoing service as defined in clause 4.5.5, or for the purpose of NAS signalling connection establishment following fallback indication from lower layers during an ongoing service as defined in clause 4.5.5, is mapped to the access category of the ongoing service in order to derive an RRC establishment cause, but barring checks will be skipped for this access attempt.<br>NOTE 2a: Access for the purpose of NAS signalling connection recovery during an ongoing MO IMS registration related signalling as defined in clause 4.5.5, or for the purpose of NAS signalling connection establishment following fallback indication from lower layers during an ongoing MO IMS registration related signalling as defined in clause 4.5.5, is mapped to the access category of the MO IMS registration related signalling in order to derive an RRC establishment cause, but barring checks will be skipped for this access attempt.<br>NOTE 3: If the UE selects a new SNPN, then the selected SNPN is used to check the membership; otherwise the UE uses the RSNPN or an SNPN equivalent to the RSNPN.<br>NOTE 4: This includes the 5GMM connection management procedures triggered by the UE-initiated NAS transport procedure for transporting the MO SMS.<br>NOTE 5: The UE configured for NAS signalling low priority is not supported in this release of specification.<br>NOTE 6: If the access category applicable for the access attempt is 1, then the UE shall additionally determine a second access category from the range 3 to 7. If more than one access category matches, the access category of the lowest rule number shall be chosen. The UE shall use the second access category only to derive an RRC establishment cause for the access attempt.<br>NOTE 7: Void.<br>NOTE 8: For the definition of categories a, b and c associated with access category 1, see 3GPP TS 22.261 [3]. The categories associated with access category 1 are distinct from the categories a, b and c associated with EAB (see 3GPP TS 22.011 [1A]).<br>NOTE 9: This includes:<br>a) the UE-initiated NAS transport procedure for transporting a mobile originated location<br>request;<br>b) the 5GMM connection management procedure triggered by a) above; and<br>c) NAS signalling connection recovery during an ongoing 5GC-MO-LR procedure.<br>NOTE 10: This includes:<br>a) the UE-initiated NAS transport procedure for transporting a mobile originated signalling<br>transaction towards the PCF;<br>b) the 5GMM connection management procedure triggered by a) above; and<br>c) NAS signalling connection recovery during an ongoing UE-requested policy provisioning procedure for V2XP (see 3GPP TS 24.587 [19B]). |  |  |  |
```

## 6. `23501_5.31.18_003`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.31.18  title: User Plane CIoT 5GS Optimisation
- section_path: 5 > 5.31 > 5.31.18
- merged_sections: -
- type: text  part: 3  tokens: 249

```text
[TS 23.501 | Rel-18 | §5.31.18 | User Plane CIoT 5GS Optimisation]
establishes an NGAP UE association with a new NG-RAN node different from the stored NGAP UE association, e.g. the UE initiates service request or registration procedure from a different NG-RAN node, the AMF initiates UE N2 release command towards the old NG-RAN node.
NG-RAN maintains the N3 tunnel endpoint information while a UE is in CM-IDLE with Suspend. UPF is instructed to remove DL N3 Tunnel Info of AN during Connection Suspend procedure, while UPF keeps UL N3 Tunnel Info (i.e. UPF accepts and forwards UL data). If a UE sends MO data with resume procedure, the NG-RAN can send the MO data to the UPF which is addressed by the N3 tunnel endpoint information. In the case of change of serving NG-RAN node due to UE mobility, if NG-RAN determines that it is not able to connect to the UPF which is addressed by the N3 tunnel endpoint information, NG-RAN performs Path Switch procedure before sending the MO data received from the UE.
Early Data Transmission may be initiated by the UE for mobile originated User Plane CIoT 5GS Optimisation when the RAT Type is E-UTRA.
```

## 7. `23501_5.20_005`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.20  title: External Exposure of Network Capability
- section_path: 5 > 5.20
- merged_sections: -
- type: text  part: 5  tokens: 419

```text
[TS 23.501 | Rel-18 | §5.20 | External Exposure of Network Capability]
- the address of the UE as provided by the AF; this may be an IP address or a MAC address;
- the corresponding DNN and/or S-NSSAI information: this may have been provided by the AF or determined by the NEF based on the requesting AF; this is needed if the UE address is an IP address.
The NEF may provide a UE Identifier in the GPSI format of MSISDN to an authenticated and authorized AF:
- that fulfils the conditions described in clause 4.15.10A of TS 23.502 [3]; and
- that has explicitly requested a translation from the UE address to a unique UE identifier (via Nnef_UEId service) when the UE MSISDN exposure is allowed and authorized by the operator; or
- the NEF may provide an AF specific UE Identifier to the AF:
- that has explicitly requested a translation from the address of the UE to a unique UE identifier (via Nnef_UEId service); or
- that has implicitly requested a translation from the address of the UE to an AF specific UE Identifier by requesting external exposure about an individual UE identified by its address.
The AF may have its own means to maintain the AF specific UE Identifier through, e.g. an AF session. After the retrieval of an AF specific UE Identifier the AF shall not keep maintaining a mapping between this identifier and the UE IP address as this mapping may change.
The AF specific UE Identifier shall not correspond to a MSISDN; it is represented as a GPSI in the form of an External Identifier. When used as an AF specific UE identifier, the External Identifier provided by the 5GCN shall be different for different AF.
NOTE 1: This is to protect user privacy.
NOTE 2: The AF specific UE identifier is ensured to be unique across different AFs as defined in TS 23.003 [19] by configuration. Such configuration is assumed to be coordinated between the different involved entities (e.g. NEF(s) and UDM/UDR).
```

## 8. `24501_9.11.3.77_006`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §9.11.3.77  title: Paging restriction
- section_path: 9 > 9.11 > 9.11.3 > 9.11.3.77
- merged_sections: -
- type: table  part: 6  tokens: 405

```text
[TS 24.501 | Rel-18 | §9.11.3.77 | Paging restriction]
Table 9.11.3.77.1: Paging restriction information element
| Paging restriction type (bits 4 to 1 of octet 3) |  |  |  |  |
|---|---|---|---|---|
| Bits |  |  |  |  |
| 4 | 3 | 2 | 1 |  |
| 0 | 0 | 0 | 0 | reserved |
| 0 | 0 | 0 | 1 | All paging is restricted |
| 0 | 0 | 1 | 0 | All paging is restricted except for voice service |
| 0 | 0 | 1 | 1 | All paging is restricted except for specified PDU session(s) |
| 0 | 1 | 0 | 0 | All paging is restricted except for voice service and specified PDU session(s) |
| All other values are reserved. |  |  |  |  |
| Bits 5 to 8 of octet 3 are spare and shall be coded as zero.<br>PSI(x) (bits 8 to 1 of octet 4 and octet 5):<br>This field indicates the PDU session identity of the PDU session for which paging is restricted.<br>PSI(0): (bit 1 of octet 4)<br>Spare and shall be coded as zero.<br>PSI(1) – PSI(15):<br>0 indicates that paging is restricted for the PDU session associated with the PDU session identity.<br>1 indicates that paging is not restricted for the PDU session associated with the PDU session identity. |  |  |  |  |
| All bits in octet 6 to 35 are spare and shall be coded as zero, if the respective octet is included in the information element. |  |  |  |  |
```

## 9. `23501_5.16.3.10_001`

- spec: TS 23.501 (Rel-18, V18.12.0)
- section: §5.16.3.10  title: IMS Voice Service via EPS Fallback or RAT fallback in 5GS
- section_path: 5 > 5.16 > 5.16.3 > 5.16.3.10
- merged_sections: -
- type: text  part: 1  tokens: 442

```text
[TS 23.501 | Rel-18 | §5.16.3.10 | IMS Voice Service via EPS Fallback or RAT fallback in 5GS]
In order to support various deployment scenarios for obtaining IMS voice service, the UE and NG-RAN may support the mechanism to direct or redirect the UE from NG-RAN either towards E-UTRA connected to 5GC (RAT fallback) or towards EPS (E-UTRAN connected to EPC System fallback).
Following principles apply for IMS Voice Service:
- The serving AMF indicates toward the UE during the Registration procedure that IMS voice over PS session is supported.
- If a request for establishing the QoS Flow for IMS voice reaches the NG-RAN, the NG-RAN responds indicating rejection of the establishment request and the NG-RAN may trigger one of the following procedures depending on UE capabilities, N26 availability, network configuration and radio conditions:
- Redirection to EPS;
- Handover procedure to EPS;
- Redirection to E-UTRA connected to 5GC; or
- Handover to E-UTRA connected to 5GC.
- If needed, Network Provided Location Information is provided as described in clauses 4.13.6.1 and 4.13.6.2 of TS 23.502 [3].
- The ongoing IMS voice session is not impacted by a change of the IMS voice over PS session indicator from supported to unsupported (e.g. the UE receives during RAT Fallback or EPS Fallback the IMS voice over PS session indicator indicating that IMS voice over PS sessions are not supported).
NOTE: Any change in IMS voice over PS session indicator applies to new IMS sessions initiated only after the ongoing IMS voice session is terminated.
During any release of RRC connection including after EPS/RAT fallback is performed, the eNB or NG-RAN node may provide to the UE dedicated idle mode priorities for NR as defined in TS 36.331 [51] taking into account RFSP, PLMNs contained in Handover Restriction List and local operator policy. If the UE remains ECM/CM-CONNECTED after the voice call has ended, the eNB or NG-RAN node may trigger handover to NR connected to 5GC, if configured to do so, taking into account local operator policy and Handover Restriction List.
```

## 10. `24501_D.2.1.3_001`

- spec: TS 24.501 (Rel-18, V18.14.0)
- section: §D.2.1.3  title: Network-requested UE policy management procedure accepted by the UE
- section_path: D > D.2 > D.2.1 > D.2.1.3
- merged_sections: -
- type: text  part: 1  tokens: 168

```text
[TS 24.501 | Rel-18 | §D.2.1.3 | Network-requested UE policy management procedure accepted by the UE]
If all instructions included in the UE policy section management list IE were executed successfully by the UE, the UE shall:
a) create a MANAGE UE POLICY COMPLETE message including the PTI value received within the MANAGE UE POLICY COMMAND message; and
b) transport the MANAGE UE POLICY COMPLETE message using the NAS transport procedure as specified in clause 5.4.5.
Upon receipt of the MANAGE UE POLICY COMPLETE message, the PCF shall stop timer T3501. The PCF should ensure that the PTI value assigned to this procedure is not released immediately.
NOTE: The way to achieve this is implementation dependent. For example, the PCF can ensure that the PTI value assigned to this procedure is not released during the time equal to or greater than the default value of timer T3501.
```
