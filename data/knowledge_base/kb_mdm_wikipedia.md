# Mobile device management

> **Source:** [English Wikipedia](https://en.wikipedia.org/wiki/Mobile_device_management) (plain-text extract).  
> **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).  
> **Retrieved:** 2026-04-26 (MediaWiki API). Keep this notice in derivative works (BY-SA).

---

Mobile device management (MDM) is the administration of mobile devices, such as smartphones, tablet computers, and laptops.  MDM is usually implemented with the use of a third-party product that has management features for particular vendors of mobile devices. Though closely related to Enterprise Mobility Management and Unified Endpoint Management, MDM differs slightly from both: unlike MDM, EMM includes mobile information management, BYOD, mobile application management and mobile content management, whereas UEM provides device management for endpoints like desktops, printers, IoT devices, and wearables.


## Overview

MDM is typically a deployment of a combination of on-device applications and configurations, corporate policies and certificates, and backend infrastructure, for the purpose of simplifying and enhancing the IT management of end user devices. In modern corporate IT environments, the sheer number and diversity of managed devices (and user behavior) has motivated MDM solutions that allow the management of devices and users in a consistent and scalable way. The overall role of MDM is to increase device supportability, security, and corporate functionality while maintaining some user flexibility.
Many organizations administer devices and applications using MDM products/services. MDM primarily deals with corporate data segregation, securing emails, securing corporate documents on devices, enforcing corporate policies, and integrating and managing mobile devices including laptops and handhelds of various categories. MDM implementations may be either on-premises or cloud-based.
Some of the core functions of MDM include:

Ensuring that diverse user equipment is configured to a consistent standard / supported set of applications, functions, or corporate policies
Updating equipment, applications, functions, or policies in a scalable manner
Ensuring that users use applications in a consistent and supportable manner
Ensuring that equipment performs consistently
Monitoring and tracking equipment (e.g. location, status, ownership, activity)
Being able to efficiently diagnose and troubleshoot equipment remotely
MDM functionality can include over-the-air distribution of applications, data and configuration settings for all types of mobile devices, including mobile phones, smartphones, tablet computers, ruggedized mobile computers, mobile printers, mobile POS devices, etc.  Most recently laptops and desktops have been added to the list of systems supported as MDM becomes more about basic device management and less about the mobile platform itself. MDM tools are leveraged for both company-owned and employee-owned (BYOD) devices across the enterprise or mobile devices owned by consumers. Consumer Demand for BYOD is now requiring a greater effort for MDM and increased security for both the devices and the enterprise they connect to, since employers and employees have different expectations concerning the types of restrictions that should be applied to mobile devices.
By controlling and protecting the data and configuration settings of all mobile devices in a network, MDM can reduce support costs and business risks. The intent of MDM is to optimize the functionality and security of a mobile communications network while minimizing cost and downtime.
With mobile devices becoming ubiquitous and applications flooding the market, mobile monitoring is growing in importance. The use of mobile device management across continues to grow at a steady pace, and is likely to register a compound annual growth rate (CAGR) of nearly 23% through 2028. The US will continue to be the largest market for mobile device management globally. Numerous vendors help mobile device manufacturers, content portals and developers test and monitor the delivery of their mobile content, applications, and services. This testing of content is done in real time by simulating the actions of thousands of customers and detecting and correcting bugs in the applications.


## Implementation

Typically solutions include a server component, which sends out the management commands to the mobile devices, and a client component, which runs on the managed device and receives and implements the management commands. In some cases, a single vendor provides both the client and the server, while in other cases the client and server come from different sources.
The management of mobile devices has evolved over time. At first, it was necessary to either connect to the handset or install a SIM in order to make changes and updates; scalability was a problem.
One of the next steps was to allow a client-initiated update, similar to when a user requests a Windows Update.
Central remote management, using commands sent over the air, is the next step. An administrator at the mobile operator, an enterprise IT data center, or a handset OEM can use an administrative console to update or configure any one handset, group, or groups of handsets. This provides scalability benefits particularly useful when the fleet of managed devices is large in size.
Device management software platforms ensure that end-users benefit from plug and play data services for whatever device they are using. Such a platform can automatically detect devices in the network, sending them settings for immediate and continued usability. The process is fully automated, keeps a history of used devices, and sends settings only to subscriber devices which were not previously set, sometimes at speeds reaching 50 over-the-air settings update files per second. The multiple application support requirements fulfilled through multi-app mode.


## Device management specifications

The Open Mobile Alliance (OMA) specified a platform-independent device management protocol called OMA Device Management. The specification meets the common definitions of an open standard, meaning the specification is freely available and implementable. It is supported by several mobile devices, such as PDAs and mobile phones.
Smart message is a text SMS-based provisioning protocol (ringtones, calendar entries but service settings also supported like ftp, telnet, SMSC number, email settings, etc...) 
OMA Client Provisioning is a binary SMS-based service settings provisioning protocol.
Nokia-Ericsson OTA is a binary SMS-based service settings provisioning protocol, designed mainly for older Nokia and Ericsson mobile phones.
Over-the-air programming (OTA) capabilities are considered the main component of mobile network operator and enterprise-grade mobile device management software. These include the ability to remotely configure a single mobile device, an entire fleet of mobile devices or any IT-defined set of mobile devices; send software and OS updates; remotely lock and wipe a device, which protects the data stored on the device when it is lost or stolen; and remote troubleshooting. OTA commands are sent as a binary SMS message. Binary SMS is a message including binary data.
Mobile device management software enables corporate IT departments to manage the many mobile devices used across the enterprise; consequently, over-the-air capabilities are in high demand. Enterprises using OTA SMS as part of their MDM infrastructure demand high quality in the sending of OTA messages, which imposes on SMS gateway providers a requirement to offer a high level of quality and reliability.


## Use in enterprise

As the bring your own device (BYOD) approach becomes increasingly popular across mobile service providers, MDM lets corporations provide employees with access to the internal networks using a device of their choice, whilst these devices are managed remotely with minimal disruption to employees' schedules.


## For mobile security

All MDM products are built with an idea of Containerization. The MDM Container is secured using the latest cryptographic techniques (AES-256 or more preferred). Corporate data such as email, documents, and enterprise applications are encrypted and processed inside the container. This ensures that corporate data is separated from the user's personal data on the device. Additionally, encryption for the entire device and/or SD Card can be enforced depending on MDM product capability.
Secure email: MDM products allow organizations to integrate their existing email setup to be easily integrated with the MDM environment. Almost all MDM products support easy integration with Exchange Server (2003/2007/2010), Office365, Lotus Notes, BlackBerry Enterprise Server (BES), and others. This provides the flexibility of configuring email over the air.
Secure docs: Employees frequently copy attachments downloaded from corporate email to their personal devices and then misuse it. MDM can restrict or disable clipboard usage into or out of the secure container, restrict the forwarding of attachments to external domains, or prevent saving attachments on the SD card. This ensures corporate data is secure.
Secure browser: Using a secure browser can avoid many potential security risks. Every MDM solution comes with a built-in custom browser. An administrator can disable native browsers to force users to use the secure browser inside the MDM container. URL filtering can be enforced to add additional security measures.
Secure app catalog: Organizations can distribute, manage, and upgrade applications on an employee's device using an App Catalog. This allows applications to be pushed onto the user's device directly from the App Store or push an enterprise developed private application through the App Catalog. This provides an option for the organization to deploy devices in Kiosk Mode or Lock-Down Mode.


## Additional MDM features

There are plenty of other features depending on which MDM product is chosen:

Policy Enforcing: There are multiple types of policies that can be enforced on MDM users.
Personal Policy: According to the corporate environment, highly customizable
Device Platform specific: policies for advanced management of Android, iOS, Windows, and Blackberry devices.
Compliance Policies/Rules
VPN configuration
Application Catalogue
Pre-defined Wi-Fi and Hotspot settings
Jailbreak/Root detection
Remote Wipe of corporate data
Remote Wipe of entire device
Device remote locking
Remote messaging/buzz
Disabling native apps on device
Some Kiosk software features


## SaaS versus on-premises solutions

Mobile device management (MDM) solutions use both software as a service (SaaS) and on-premises deployment models. SaaS systems are cloud-based, while on-premises systems use physical or virtual machines and require software maintenance.
For security in cloud computing, the US Government has compliance audits such as Federal Information Security Management Act of 2002 (FISMA) which cloud providers can go through to meet security standards.
The primary policy approach taken by Federal agencies to build relationships with cloud service providers is Federal Risk and Authorization Management Program (FedRAMP) accreditation and certification, designed in part to protect FISMA Low, Moderate, High and Li-SaaS systems.


## Evolution of MDM

MDM is also about managing the device features, but its coupled with mobile content management (MCM) and Mobile Identity Management (MIM), Application management (MAM) is referred to as Enterprise Mobility Management (EMM). As EMM was specifically about managing the apps and content on mobile devices it was not able to manage older devices such as Windows laptops/desktops and new Macs, so EMM evolved into UEM (Unified Endpoint Management) with additional functionality to manage mobile, tablet and traditional devices such as desktops and laptops.


## See also

The SyncML Initiative
OMA Device Management
Open Mobile Alliance
Over-the-air programming
Mobile application management
Mobile content management system
Mobile security
Enterprise mobility management
BlackBerry Enterprise Server
List of Mobile Device Management software
Unified Endpoint Management


== References ==