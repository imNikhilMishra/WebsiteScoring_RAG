import os
import json
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# === Replace with your actual data ===
# css_selectors = [
#     "selector-1", "selector-2", "selector-3"
# ]
# html_components = [
#     "<div>Some HTML</div>", "<div>Some more HTML</div>", "<div>Another component</div>"
# ]
# image_paths = [
#     "images/component-01.png",
#     "images/component-02.png",
#     "images/component-03.png"
# ]

css_selectors = [
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-1.dnd-section.dnd_area-row-0-padding",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-6.dnd_area-row-1-padding.dnd_area-row-1-background-color.dnd_area-row-1-max-width-section-centering.dnd-section.dnd_area-row-1-background-layers",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-9.dnd-section",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-15.dnd_area-row-3-max-width-section-centering.dnd-section.dnd_area-row-3-background-layers.dnd_area-row-3-background-color",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-18.dnd-section.dnd_area-row-4-background-layers.dnd_area-row-4-background-color.dnd_area-row-4-vertical-alignment",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-22.dnd-section.dnd_area-row-5-background-layers.dnd_area-row-5-vertical-alignment.dnd_area-row-5-background-color",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-26.dnd_area-row-6-vertical-alignment.dnd-section.dnd_area-row-6-max-width-section-centering.dnd_area-row-6-background-image.dnd_area-row-6-background-layers",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-30.dnd-section",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-33.dnd-section.dnd_area-row-8-background-layers.dnd_area-row-8-background-color",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-36.dnd-section.dnd_area-row-9-force-full-width-section.dnd_area-row-9-background-color.dnd_area-row-9-background-layers",
    "#main-content > div > div > div > div > div.row-fluid-wrapper.row-depth-1.row-number-40.dnd-section.dnd_area-row-10-background-layers.dnd_area-row-10-background-color",
    "#gsi-global-footer"
]

html_components = [
    # 1
    """<div class="row-fluid-wrapper row-depth-1 row-number-1 dnd-section dnd_area-row-0-padding">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column" data-widget-type="cell" data-x="0" data-w="12">
      <div class="row-fluid-wrapper row-depth-1 row-number-2 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget module_17171843573815-flexbox-positioning dnd-module">
            <div id="hs_cos_wrapper_module_17171843573815" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-linked_image">
              <span id="hs_cos_wrapper_module_17171843573815_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_linked_image">
                <img src="https://www.getgsi.com/hs-fs/hubfs/GSI_Logo_PMS%20cloud.png?width=200&amp;height=137&amp;name=GSI_Logo_PMS%20cloud.png"
                     class="hs-image-widget" width="200" height="137" style="max-width: 100%; height: auto;"
                     alt="GSI_Logo_PMS cloud" title="GSI_Logo_PMS cloud" loading="lazy"
                     srcset="
                       https://www.getgsi.com/hs-fs/hubfs/GSI_Logo_PMS%20cloud.png?width=100&amp;height=69&amp;name=GSI_Logo_PMS%20cloud.png 100w,
                       https://www.getgsi.com/hs-fs/hubfs/GSI_Logo_PMS%20cloud.png?width=200&amp;height=137&amp;name=GSI_Logo_PMS%20cloud.png 200w,
                       https://www.getgsi.com/hs-fs/hubfs/GSI_Logo_PMS%20cloud.png?width=300&amp;height=206&amp;name=GSI_Logo_PMS%20cloud.png 300w,
                       https://www.getgsi.com/hs-fs/hubfs/GSI_Logo_PMS%20cloud.png?width=400&amp;height=274&amp;name=GSI_Logo_PMS%20cloud.png 400w,
                       https://www.getgsi.com/hs-fs/hubfs/GSI_Logo_PMS%20cloud.png?width=500&amp;height=343&amp;name=GSI_Logo_PMS%20cloud.png 500w,
                       https://www.getgsi.com/hs-fs/hubfs/GSI_Logo_PMS%20cloud.png?width=600&amp;height=411&amp;name=GSI_Logo_PMS%20cloud.png 600w"
                     sizes="(max-width: 200px) 100vw, 200px">
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-3 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_17171843573816" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_module_17171843573816_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h1 style="text-align: center; font-size: 48px;">
                  <span style="color: #222222;">Future-Proof Your Business with GSI</span>
                </h1>
                <h4 style="font-size: 24px; text-align: center;">
                  <span style="color: #101c3b;">JD Edwards, NetSuite, HubSpot, and Infrastructure Technologies</span>
                </h4>
                <p style="font-size: 24px; text-align: center;">
                  <span style="color: #101c3b;">
                    Alongside these enterprise platforms, GSI's world-class consultants offer AI, cloud, data, and cybersecurity services, empowering your business to thrive in a digital-first world.
                  </span>
                </p>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-4 dnd-row cell_1717184357381-row-2-vertical-alignment">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-cell cell_17171843573817-vertical-alignment dnd-column">
            <div class="row-fluid-wrapper row-depth-1 row-number-5 dnd-row">
              <div class="row-fluid ">
                <div class="span12 widget-span widget-type-custom_widget dnd-module">
                  <div id="hs_cos_wrapper_module_17171843573819" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
                    <div class="gsi-cta-module desktop-center mobile-center col-cta-1">
                      <a href="https://www.getgsi.com/contact-us/?hsLang=en"
                         class="gsi-cta-insie"
                         title="Schedule a Consultation"
                         target="_blank" rel="noopener">
                        <div>Schedule a Consultation</div>
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 2
    """<div class="row-fluid-wrapper row-depth-1 row-number-6 dnd_area-row-1-padding dnd_area-row-1-background-color dnd_area-row-1-max-width-section-centering dnd-section dnd_area-row-1-background-layers">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-7 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_17171843696226" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_module_17171843696226_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2 style="text-align: center;">
                  <span style="color: #ffffff;">
                    Transform Your Business Landscape With our Integrated Suite of Enterprise Applications and Services
                  </span>
                </h2>
                <p style="text-align: center;">
                  <span style="color: #ffffff;">
                    When you work with GSI you'll access our unique expertise as a leading solution provider for the leading enterprise applications, plus access to powerful cloud and data solutions. Get more information on all of our available solutions by clicking below:
                  </span>
                </p>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-8 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_17171843696227" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <section id="gsi-industries">
                <div class="gsi-industries-container">
                  <a href="https://www.getgsi.com/jd-edwards/managed-services/?hsLang=en" class="industry" title="Explore JD Edwards Solutions">
                    <div>
                      <div class="gsi-industries-image">
                        <img src="https://www.getgsi.com/hubfs/GSI%20Website%20Assets/JD%20Edwards%20Support%20Services%20by%20GSIO-1.webp"
                             alt="JD Edwards Support Services by GSIO-1">
                      </div>
                      <div class="gsi-industries">
                        <div>Explore JD Edwards Solutions</div>
                      </div>
                    </div>
                  </a>
                  <a href="https://www.getgsi.com/netsuite?hsLang=en" class="industry" title="Power Up with NetSuite">
                    <div>
                      <div class="gsi-industries-image">
                        <img src="https://www.getgsi.com/hubfs/quickbooks%20vs%20netsuite.png"
                             alt="quickbooks vs netsuite">
                      </div>
                      <div class="gsi-industries">
                        <div>Power Up with NetSuite</div>
                      </div>
                    </div>
                  </a>
                  <a href="https://www.getgsi.com/hubspot?hsLang=en" class="industry" title="Unlock HubSpot Excellence">
                    <div>
                      <div class="gsi-industries-image">
                        <img src="https://www.getgsi.com/hubfs/Flawless%20Inbound%20Website%20Assets/Imported_Blog_Media/hubspot.png"
                             alt="hubspot">
                      </div>
                      <div class="gsi-industries">
                        <div>Unlock HubSpot Excellence</div>
                      </div>
                    </div>
                  </a>
                  <a href="https://www.getgsi.com/servicenow-consulting-services?hsLang=en" class="industry" title="Streamline with ServiceNow">
                    <div>
                      <div class="gsi-industries-image">
                        <img src="https://www.getgsi.com/hubfs/GSI%20Website%20Assets/managed%20services-1.webp"
                             alt="managed services-1">
                      </div>
                      <div class="gsi-industries">
                        <div>Streamline with ServiceNow</div>
                      </div>
                    </div>
                  </a>
                  <a href="https://www.getgsi.com/jd-edwards/products/genius-ai/?hsLang=en" class="industry" title="Unleash AI Potential">
                    <div>
                      <div class="gsi-industries-image">
                        <img src="https://www.getgsi.com/hubfs/Flawless%20Inbound%20Website%20Assets/Imported_Blog_Media/appmode_AI_Image_previmageUPDATE-1.jpg"
                             alt="appmode_AI_Image_previmageUPDATE-1">
                      </div>
                      <div class="gsi-industries">
                        <div>Unleash AI Potential</div>
                      </div>
                    </div>
                  </a>
                  <a href="https://www.getgsi.com/cloud/cloud-overview/?hsLang=en" class="industry" title="Discover Cloud Services">
                    <div>
                      <div class="gsi-industries-image">
                        <img src="https://www.getgsi.com/hubfs/GSI%20Website%20Assets/cloud%20technology.webp"
                             alt="cloud technology">
                      </div>
                      <div class="gsi-industries">
                        <div>Discover Cloud Services</div>
                      </div>
                    </div>
                  </a>
                  <a href="https://www.getgsi.com/cybersecurity-services/?hsLang=en" class="industry" title="Stay Secure with Cybersecurity">
                    <div>
                      <div class="gsi-industries-image">
                        <img src="https://www.getgsi.com/hubfs/cybersecurity%20(1)%20copy.png"
                             alt="cybersecurity (1) copy">
                      </div>
                      <div class="gsi-industries">
                        <div>Stay Secure with Cybersecurity</div>
                      </div>
                    </div>
                  </a>
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 3
    """<div class="row-fluid-wrapper row-depth-1 row-number-9 dnd-section">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-10 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_17171843816026" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_module_17171843816026_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h3 style="font-size: 36px; text-align: center;">
                  <span style="color: #2d3750;">Rave Reviews: Discover Why Our Customers Are Giving Us 5 Stars</span>
                </h3>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-11 dnd-row">
        <div class="row-fluid ">
          <div class="span4 widget-span widget-type-cell dnd-column">
            <div class="row-fluid-wrapper row-depth-1 row-number-12 dnd-row">
              <div class="row-fluid ">
                <div class="span12 widget-span widget-type-custom_widget dnd-module">
                  <div id="hs_cos_wrapper_module_171718438160211" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-text">
                    <span id="hs_cos_wrapper_module_171718438160211_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_text">
                      <review-widget widget-id="0bdca87e-f819-4a40-b1ed-ca002080b45f"></review-widget>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="span4 widget-span widget-type-cell dnd-column">
            <div class="row-fluid-wrapper row-depth-1 row-number-13 dnd-row">
              <div class="row-fluid ">
                <div class="span12 widget-span widget-type-custom_widget dnd-module">
                  <div id="hs_cos_wrapper_module_171718438160213" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-text">
                    <span id="hs_cos_wrapper_module_171718438160213_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_text">
                      <review-widget widget-id="72d0612e-6fc8-4409-ae23-b5a37b535060"></review-widget>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="span4 widget-span widget-type-cell dnd-column">
            <div class="row-fluid-wrapper row-depth-1 row-number-14 dnd-row">
              <div class="row-fluid ">
                <div class="span12 widget-span widget-type-custom_widget dnd-module">
                  <div id="hs_cos_wrapper_module_171718438160215" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-text">
                    <span id="hs_cos_wrapper_module_171718438160215_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_text">
                      <review-widget widget-id="cbad8090-7ebf-447a-b0a1-302c835fb39a"></review-widget>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 4
    """<div class="row-fluid-wrapper row-depth-1 row-number-15 dnd_area-row-3-max-width-section-centering dnd-section dnd_area-row-3-background-layers dnd_area-row-3-background-color">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-16 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1674835179827" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_widget_1674835179827_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2 style="text-align: center;">Tomorrow's Companies Working with GSI</h2>
                <p style="text-align: center;">
                  We work with forward-thinking organizations looking to align and optimize their digital footprint with their business goals.
                </p>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-17 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1705524385826" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <section id="gsi-industries">
                <div class="gsi-industries-container">
                  <!-- ... industry links as above ... -->
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 5
    """<div class="row-fluid-wrapper row-depth-1 row-number-18 dnd-section dnd_area-row-4-background-layers dnd_area-row-4-background-color dnd_area-row-4-vertical-alignment">
  <div class="row-fluid ">
    <div class="span6 widget-span widget-type-cell cell_1675277237284-vertical-alignment dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-19 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1675277237107" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_widget_1675277237107_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2><span>Get Enterprise Applications with GSI</span></h2>
                <p><span>We are an Oracle Platinum Partner and authorized reseller of <a href="/netsuite?hsLang=en">Oracle NetSuite</a>, <a href="/jd-edwards/enterpriseone/?hsLang=en">Oracle JD Edwards</a>, <a href="/servicenow-consulting-services?hsLang=en">ServiceNow</a> and <a href="/hubspot?hsLang=en">HubSpot</a>. We take a holistic look at your business challenge and partner with you to apply the best solution, regardless of the source.</span></p>
                <span>Elevate your business with our <a href="/application-modernization/?hsLang=en">application modernization</a> and integration services, transforming legacy systems and seamlessly connecting applications for streamlined operations and accelerated growth.</span>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-20 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_1682020898281" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="gsi-cta-module desktop-left mobile-center col-cta-1">
                <a href="/enterprise-applications?hsLang=en" class="gsi-cta-insie" title="Learn More">
                  <div>Learn More</div>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="span6 widget-span widget-type-cell cell_1705524280483-vertical-alignment dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-21 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1705524148081" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="simple-video">
                <video width="100%" controls poster="https://44161995.fs1.hubspotusercontent-na1.net/hubfs/44161995/Enterprise%20Applications-min.png">
                  <source src="https://www.getgsi.com/hubfs/GSI%20Website%20Assets/GSI%20Videos/GSI%20Enterprise%20Applications%20Overview%20video.mp4" type="video/mp4">
                  Your browser does not support this video format.
                </video>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 6
    """<div class="row-fluid-wrapper row-depth-1 row-number-22 dnd-section dnd_area-row-5-background-layers dnd_area-row-5-vertical-alignment dnd_area-row-5-background-color">
  <div class="row-fluid ">
    <div class="span6 widget-span widget-type-cell cell_1675278109795-vertical-alignment dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-23 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_17055242668093" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="simple-video">
                <video width="100%" controls poster="https://44161995.fs1.hubspotusercontent-na1.net/hubfs/44161995/Cloud%20Services.png">
                  <source src="https://www.getgsi.com/hubfs/GSI%20Website%20Assets/GSI%20Videos/JD%20Edwards%20Cloud%20Services%20Solutions.mp4" type="video/mp4">
                  Your browser does not support this video format.
                </video>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="span6 widget-span widget-type-cell cell_16748381477672-vertical-alignment dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-24 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_16748381451233" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <span id="hs_cos_wrapper_module_16748381451233_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2>GET Cloud Services with GSI</h2>
                <p>We offer an extensive array of cloud/hosting options (public, private, hybrid, and multi-cloud) to optimize your cloud strategy, roadmap, migration, implementation, and managed services. Migrating to the cloud has never been easier.</p>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-25 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1675232510071" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="gsi-cta-module desktop-left mobile-center col-cta-1">
                <a href="/cloud/cloud-overview/?hsLang=en" class="gsi-cta-insie" title="Learn More">
                  <div>Learn More</div>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 7
    """<div class="row-fluid-wrapper row-depth-1 row-number-26 dnd_area-row-6-vertical-alignment dnd-section dnd_area-row-6-max-width-section-centering dnd_area-row-6-background-image dnd_area-row-6-background-layers">
  <div class="row-fluid ">
    <div class="span6 widget-span widget-type-cell cell_1674838450910-vertical-alignment dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-27 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_16748384482798" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <span id="hs_cos_wrapper_module_16748384482798_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2>GET Cybersecurity Services with GSI</h2>
                <p>Does your organization have an effective plan in place to protect its critical assets from threats? Have you identified and addressed security vulnerabilities? GSI provides a broad range of security offerings to support your critical cybersecurity needs, starting with our security assessment and remediation plan.</p>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-28 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1675273247345" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="gsi-cta-module desktop-left mobile-center col-cta-1">
                <a href="/cybersecurity-services/?hsLang=en" class="gsi-cta-insie" title="Learn More">
                  <div>Learn More</div>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="span6 widget-span widget-type-cell cell_1705524290836-vertical-alignment dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-29 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_17055242882983" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="simple-video">
                <video width="100%" controls poster="https://44161995.fs1.hubspotusercontent-na1.net/hubfs/44161995/GSI%20Cybersecurity%20Services-1.png">
                  <source src="https://www.getgsi.com/hubfs/GSI%20Website%20Assets/GSI%20Videos/Cybersecurity%20Overview_UPDATE_FINAL.m4v" type="video/mp4">
                  Your browser does not support this video format.
                </video>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 8
    """<div class="row-fluid-wrapper row-depth-1 row-number-30 dnd-section">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-31 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1674838783346" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_widget_1674838783346_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2 style="text-align: center;">Business Process Optimization with GSI</h2>
                <p style="text-align: center;">Utilizing our extensive experience and knowledge of Enterprise Applications, we analyze the usage of your application portfolio to ensure a maximum value is extracted from your business. This is achieved through our Business Value Assessments where we assist you in building your roadmap to your future state digital footprint to help you maximize your systems investment.</p>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-32 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1675273265049" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="gsi-cta-module desktop-center mobile-center col-cta-1">
                <a href="https://www.getgsi.com/business-process-optimization?hsLang=en" class="gsi-cta-insie" title="Learn More">
                  <div>Learn More</div>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 9
    """<div class="row-fluid-wrapper row-depth-1 row-number-33 dnd-section dnd_area-row-8-background-layers dnd_area-row-8-background-color">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-34 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_1674839075054" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_module_1674839075054_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2 style="text-align: center;">Certifications & Awards</h2>
                <p style="text-align: center;">GSI offers a broad range of ERP consulting services that utilize experts with deep industry knowledge, extensive global implementation experience and numerous certifications and awards.</p>
                <h3 style="text-align: center; margin: 0;">Awards</h3>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-35 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1705517897165" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <section class="customer-logos slider slick-initialized slick-slider">
                <!-- carousel markup omitted for brevity -->
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 10
    """<div class="row-fluid-wrapper row-depth-1 row-number-36 dnd-section dnd_area-row-9-force-full-width-section dnd_area-row-9-background-color dnd_area-row-9-background-layers">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-37 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_17171844050886" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <section id="two-quotes">
                <!-- quotes markup omitted for brevity -->
              </section>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-38 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-cell dnd-column cell_17171844050887-padding">
            <div class="row-fluid-wrapper row-depth-1 row-number-39 dnd-row">
              <div class="row-fluid ">
                <div class="span12 widget-span widget-type-custom_widget dnd-module">
                  <div id="hs_cos_wrapper_module_17171844050889" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
                    <div class="gsi-cta-module desktop-center mobile-center col-cta-2">
                      <a href="https://www.glassdoor.com/Reviews/GSI-Reviews-E845455.htm" class="gsi-cta-insie" title="See Glassdoor Reviews">
                        <div>See Glassdoor Reviews</div>
                      </a>
                      <a href="https://www.getgsi.com/company-overview?hsLang=en" class="gsi-cta-insie" title="About GSI">
                        <div>About GSI</div>
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 11
    """<div class="row-fluid-wrapper row-depth-1 row-number-40 dnd-section dnd_area-row-10-background-layers dnd_area-row-10-background-color">
  <div class="row-fluid ">
    <div class="span12 widget-span widget-type-cell dnd-column">
      <div class="row-fluid-wrapper row-depth-1 row-number-41 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_module_16748390960249" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module widget-type-rich_text">
              <span id="hs_cos_wrapper_module_16748390960249_" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_rich_text">
                <h2 style="text-align: center;"><span style="color: #ffffff;">Ready To Start?</span></h2>
                <p style="text-align: center;"><span style="color: #ffffff;">Our mission is to make every customer a client by offering competitively-priced, full-customizable products and services, providing only the most experienced consultants, and delivering the highest level of service day-after-day, year-after-year.</span></p>
              </span>
            </div>
          </div>
        </div>
      </div>
      <div class="row-fluid-wrapper row-depth-1 row-number-42 dnd-row">
        <div class="row-fluid ">
          <div class="span12 widget-span widget-type-custom_widget dnd-module">
            <div id="hs_cos_wrapper_widget_1679426708484" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module">
              <div class="gsi-cta-module desktop-center color-variation-blue mobile-center col-cta-1">
                <a href="/contact-us/?hsLang=en" class="gsi-cta-insie" title="Schedule a Free Consultation">
                  <div>Schedule a Free Consultation</div>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""",
    # 12
    """<section id="gsi-global-footer">
  <!-- global footer full HTML as above -->
</section>"""
]

image_paths = [
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_1_dnd_section_dnd_area_row_0_padding.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_6_dnd_area_row_1_padding_dnd_area_row_1_background_color_dnd_area_row_1_max_width_section_centering_dnd_section_dnd_are.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_9_dnd_section.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_15_dnd_area_row_3_max_width_section_centering_dnd_section_dnd_area_row_3_background_layers_dnd_area_row_3_background_co.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_18_dnd_section_dnd_area_row_4_background_layers_dnd_area_row_4_background_color_dnd_area_row_4_vertical_alignment.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_22_dnd_section_dnd_area_row_5_background_layers_dnd_area_row_5_vertical_alignment_dnd_area_row_5_background_color.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_26_dnd_area_row_6_vertical_alignment_dnd_section_dnd_area_row_6_max_width_section_centering_dnd_area_row_6_background_i.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_30_dnd_section.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_33_dnd_section_dnd_area_row_8_background_layers_dnd_area_row_8_background_color.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_36_dnd_section_dnd_area_row_9_force_full_width_section_dnd_area_row_9_background_color_dnd_area_row_9_background_layers.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__main_content___div___div___div___div___div_row_fluid_wrapper_row_depth_1_row_number_40_dnd_section_dnd_area_row_10_background_layers_dnd_area_row_10_background_color.png",
    "D:\\RAG_SpikeAI\\RAG_main\\images\\component__gsi_global_footer.png"
]


# === Model loading ===
text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")

def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator="\n", strip=True)

def compute_image_embedding(image_path):
    print(f"Trying to open: {image_path}")
    print(f"File exists? {os.path.exists(image_path)}")
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
    with torch.no_grad():
        return clip_model.encode_image(image).squeeze().tolist()

def compute_text_embedding(text):
    return text_model.encode(text).tolist()

# === Build KB ===
knowledge_base = []
for idx, (selector, html, image_path) in enumerate(zip(css_selectors, html_components, image_paths), start=1):
    text_content = extract_text(html)
    text_embedding = compute_text_embedding(text_content)
    image_embedding = compute_image_embedding(image_path)

    kb_entry = {
        "component_id": f"component-{idx:02}",
        "page_url": "https://www.getgsi.com/",
        "css_selector": selector,
        "html": html,
        "text_content": text_content,
        "screenshot_path": image_path,
        "text_embedding": text_embedding,
        "image_embedding": image_embedding
    }

    knowledge_base.append(kb_entry)

# === Save JSON ===
with open("getgsi_knowledge_base.json", "w") as f:
    json.dump(knowledge_base, f, indent=2)

print("✅ Knowledge Base saved to getgsi_knowledge_base.json")
