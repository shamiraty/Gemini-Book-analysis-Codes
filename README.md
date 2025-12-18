```python
!pip install PyPDF2 pandas openpyxl
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting PyPDF2
      Using cached pypdf2-3.0.1-py3-none-any.whl (232 kB)
    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (1.5.3)
    Requirement already satisfied: openpyxl in c:\programdata\anaconda3\lib\site-packages (3.0.10)
    Requirement already satisfied: numpy>=1.21.0 in c:\programdata\anaconda3\lib\site-packages (from pandas) (1.23.5)
    Requirement already satisfied: pytz>=2020.1 in c:\users\samir\appdata\roaming\python\python310\site-packages (from pandas) (2025.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: et_xmlfile in c:\programdata\anaconda3\lib\site-packages (from openpyxl) (1.1.0)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)
    Installing collected packages: PyPDF2
    Successfully installed PyPDF2-3.0.1
    


```python
"""
COMPREHENSIVE BOOK ANALYZER
============================
Analyzes books (PDF format) using Google's Gemini AI with detailed metadata extraction,
thematic analysis, and comparative review.

Features:
- Complete bibliographic metadata extraction
- Thematic analysis and content categorization
- Writing style and approach analysis
- Target audience identification
- Comparative analysis across multiple books
- Professional Markdown report generation

Author: Shamir
Date: 2024
Folder: BOOKS2
"""

import google.generativeai as genai
import pandas as pd
from PyPDF2 import PdfReader
import os
from typing import List, Dict
import json
import time
from datetime import datetime
from collections import Counter


class BookAnalyzer:
    """Comprehensive analyzer for books using Gemini AI"""
    
    def __init__(self, api_key: str):
        """
        Initialize the book analyzer with Gemini API key
        
        Args:
            api_key (str): Google Gemini API key
        """
        print(f"🔑 Configuring API key...")
        genai.configure(api_key=api_key)
        
        # Verify API connection
        try:
            models = list(genai.list_models())
            print(f"✅ API connection successful! Available models: {len(models)}")
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            raise
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.books_data = []  # Storage for all extracted book data
        
        print(f"✅ Using model: gemini-2.5-flash")
        print(f"⚠️  FREE PLAN MODE: 20-second delays between API calls\n")
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 30) -> str:
        """
        Extract text content from PDF file
        
        Args:
            pdf_path (str): Full path to PDF file
            max_pages (int): Maximum pages to read (default: 30)
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"   → Total pages in book: {total_pages}")
                
                # Limit pages to read (saves time and tokens)
                pages_to_read = min(max_pages, total_pages)
                extracted_text = ''
                
                print(f"   → Extracting text from first {pages_to_read} pages...")
                
                # Extract text page by page with markers
                for page_num in range(pages_to_read):
                    page_content = pdf_reader.pages[page_num].extract_text()
                    extracted_text += f"\n=== PAGE {page_num + 1} ===\n{page_content}"
                
                print(f"   ✅ Extracted {len(extracted_text)} characters")
                return extracted_text.strip()
                
        except Exception as e:
            print(f"   ❌ PDF extraction error: {e}")
            return ""
    
    def extract_book_metadata(self, book_text: str, filename: str) -> Dict:
        """
        Extract comprehensive metadata from book text using Gemini AI
        
        Args:
            book_text (str): Extracted text from the book
            filename (str): Name of the PDF file
            
        Returns:
            Dict: Comprehensive dictionary of book metadata
        """
        # Limit text to avoid token limits (15,000 chars ≈ 4,000 tokens)
        text_sample = book_text[:15000]
        
        # Create comprehensive prompt for book analysis
        prompt = f"""Analyze this book thoroughly and extract detailed information in JSON format.

BOOK TEXT:
{text_sample}

Extract the following information (be as detailed and accurate as possible):

**BIBLIOGRAPHIC INFORMATION:**
1. "title": Complete book title (including subtitle if present)
2. "author": Full author name(s) - include all authors
3. "publication_year": Year of publication
4. "publisher": Publisher name
5. "publication_place": City and country of publication
6. "edition": Edition number (e.g., "1st", "2nd", "Revised")
7. "isbn": ISBN number if available
8. "pages": Total number of pages if mentioned

**CONTENT ANALYSIS:**
9. "book_type": Type of book (e.g., "Textbook", "Novel", "Reference", "Manual", "Biography")
10. "subject_area": Main subject area (e.g., "Computer Science", "Medicine", "Business")
11. "book_summary": Comprehensive summary (3-5 sentences describing main content)
12. "main_themes": Array of 5-8 main themes/topics covered
13. "key_concepts": Array of important concepts or ideas presented
14. "target_audience": Who is this book for? (e.g., "Undergraduate students", "Professionals", "General readers")
15. "language": Primary language of the book
16. "difficulty_level": Difficulty level (e.g., "Beginner", "Intermediate", "Advanced")

**STRUCTURE & STYLE:**
17. "table_of_contents": Major chapter titles or sections (if available)
18. "writing_style": Description of writing style (e.g., "Academic", "Conversational", "Technical")
19. "approach": Teaching/presentation approach (e.g., "Theoretical", "Practical", "Case-study based")
20. "special_features": Notable features (e.g., "Includes exercises", "Has illustrations", "Contains code examples")

**ACADEMIC INFORMATION:**
21. "foreword_by": Who wrote the foreword (if applicable)
22. "preface_summary": Brief summary of preface/introduction
23. "intended_use": How the book is meant to be used (e.g., "Course textbook", "Self-study", "Reference")

**CITATIONS:**
24. "citation_apa": Complete APA format citation
25. "citation_mla": Complete MLA format citation
26. "citation_chicago": Chicago style citation

**ADDITIONAL METADATA:**
27. "copyright_year": Copyright year if different from publication year
28. "series_info": If part of a series, mention series name
29. "related_fields": Related academic or professional fields

Return ONLY valid JSON without any markdown formatting.
If information is not found, use "Unknown" for strings or empty array [] for lists.

Example format:
{{
    "title": "Introduction to Machine Learning",
    "author": "John Smith, Jane Doe",
    "publication_year": "2023",
    "publisher": "Tech Press",
    "publication_place": "Boston, USA",
    "edition": "2nd",
    "isbn": "978-0-123456-78-9",
    "pages": "450",
    "book_type": "Textbook",
    "subject_area": "Computer Science",
    "book_summary": "This textbook provides a comprehensive introduction to machine learning...",
    "main_themes": ["supervised learning", "neural networks", "deep learning"],
    "key_concepts": ["classification", "regression", "model training"],
    "target_audience": "Undergraduate and graduate students",
    "language": "English",
    "difficulty_level": "Intermediate",
    "table_of_contents": ["Introduction", "Supervised Learning", "Neural Networks"],
    "writing_style": "Academic but accessible",
    "approach": "Theory with practical examples",
    "special_features": "Includes Python code examples and exercises",
    "foreword_by": "Unknown",
    "preface_summary": "The book aims to provide hands-on experience...",
    "intended_use": "University course textbook",
    "citation_apa": "Smith, J., & Doe, J. (2023). Introduction to Machine Learning (2nd ed.). Tech Press.",
    "citation_mla": "Smith, John, and Jane Doe. Introduction to Machine Learning. 2nd ed., Tech Press, 2023.",
    "citation_chicago": "Smith, John, and Jane Doe. Introduction to Machine Learning. 2nd ed. Boston: Tech Press, 2023.",
    "copyright_year": "2023",
    "series_info": "Computer Science Series",
    "related_fields": ["Artificial Intelligence", "Data Science", "Statistics"]
}}
"""
        
        try:
            print(f"   → Analyzing book with Gemini AI...")
            
            # Initialize response variable
            response = None
            max_retries = 5  # Allow up to 5 retry attempts
            
            # Retry loop to handle rate limiting
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    error_message = str(e).lower()
                    
                    # Check if it's a rate limit error
                    if any(keyword in error_message for keyword in ["429", "quota", "rate", "limit"]):
                        # Progressive backoff: 20s, 30s, 40s, 50s, 60s
                        wait_time = 20 + (attempt * 10)
                        print(f"   ⏳ Rate limit detected! Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        # Different error - log and raise
                        print(f"   ❌ API Error: {e}")
                        raise
            
            # Verify we got a response
            if response is None:
                print(f"   ❌ Failed to get response after {max_retries} attempts!")
                return self._create_default_metadata(filename)
            
            # Extract and clean response text
            result_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                parts = result_text.split('```')
                if len(parts) >= 2:
                    result_text = parts[1].strip()
            
            # Parse JSON response
            metadata = json.loads(result_text)
            metadata['filename'] = filename
            
            print(f"   ✅ Metadata successfully extracted!")
            
            # Mandatory delay to respect free tier rate limits
            print(f"   ⏳ Cooling down for 20 seconds...")
            time.sleep(20)
            
            return metadata
            
        except json.JSONDecodeError as json_error:
            print(f"   ⚠️ JSON parsing failed: {json_error}")
            print(f"   Raw response (first 300 chars): {result_text[:300] if result_text else 'Empty response'}...")
            return self._create_default_metadata(filename)
            
        except Exception as general_error:
            print(f"   ⚠️ Unexpected error occurred: {general_error}")
            return self._create_default_metadata(filename)
    
    def _create_default_metadata(self, filename: str) -> Dict:
        """
        Create default metadata structure when extraction fails
        
        Args:
            filename (str): Name of the PDF file
            
        Returns:
            Dict: Dictionary with default "Unknown" values for all fields
        """
        return {
            'filename': filename,
            'title': 'Unknown',
            'author': 'Unknown',
            'publication_year': 'Unknown',
            'publisher': 'Unknown',
            'publication_place': 'Unknown',
            'edition': 'Unknown',
            'isbn': 'Unknown',
            'pages': 'Unknown',
            'book_type': 'Unknown',
            'subject_area': 'Unknown',
            'book_summary': 'Could not extract summary',
            'main_themes': [],
            'key_concepts': [],
            'target_audience': 'Unknown',
            'language': 'Unknown',
            'difficulty_level': 'Unknown',
            'table_of_contents': [],
            'writing_style': 'Unknown',
            'approach': 'Unknown',
            'special_features': 'Unknown',
            'foreword_by': 'Unknown',
            'preface_summary': 'Unknown',
            'intended_use': 'Unknown',
            'citation_apa': 'Unknown',
            'citation_mla': 'Unknown',
            'citation_chicago': 'Unknown',
            'copyright_year': 'Unknown',
            'series_info': 'Unknown',
            'related_fields': []
        }
    
    def compare_books(self, books_metadata: List[Dict]) -> str:
        """
        Perform comprehensive comparative analysis of multiple books
        
        Args:
            books_metadata (List[Dict]): List of book metadata dictionaries
            
        Returns:
            str: Detailed comparative analysis in markdown format
        """
        # Prepare concise summary for comparison
        books_summary = []
        for book in books_metadata:
            books_summary.append({
                'title': book.get('title', 'Unknown'),
                'author': book.get('author', 'Unknown'),
                'year': book.get('publication_year', 'Unknown'),
                'publisher': book.get('publisher', 'Unknown'),
                'book_type': book.get('book_type', 'Unknown'),
                'subject_area': book.get('subject_area', 'Unknown'),
                'summary': book.get('book_summary', 'Unknown'),
                'themes': book.get('main_themes', []),
                'target_audience': book.get('target_audience', 'Unknown'),
                'writing_style': book.get('writing_style', 'Unknown'),
                'approach': book.get('approach', 'Unknown'),
                'difficulty_level': book.get('difficulty_level', 'Unknown')
            })
        
        # Create comprehensive comparison prompt
        comparison_prompt = f"""TENGENEZA COMPREHENSIVE COMPARISON YA VITABU HIVI {len(books_metadata)}:

DATA YA VITABU:
{json.dumps(books_summary, indent=2, ensure_ascii=False)}

Toa DETAILED COMPARATIVE ANALYSIS kuhusu:

1. **MADA KUU (Main Topics & Subject Areas)**
   - Je, vitabu vyote vinahusu mada gani?
   - Ni mada zipi zinafanana? Zipi zinatofautiana kabisa?
   - Je, kuna progression au build-up ya mada kati ya vitabu?
   - Ni subject area gani zinadominate?

2. **WAANDISHI (Authors & Their Backgrounds)**
   - Kuna waandishi wangapi tofauti?
   - Je, kuna author anayeonekana mara nyingi?
   - Background za waandishi (academic, professional, etc.)
   - Credibility na expertise ya kila mwandishi

3. **MITINDO YA UANDISHI (Writing Styles & Approaches)**
   - Wametofautiana wapi kwenye style ya kuandika?
   - Ni approach gani tofauti wanazotumia? (Academic vs Practical vs Narrative)
   - Je, kuna vitabu ambavyo ni easy to read na vingine ni complex?
   - Tone ya kila kitabu (formal, conversational, technical, etc.)

4. **MIAKA YA UCHAPISHAJI (Publication Timeline & Trends)**
   - Vitabu vya zamani vs vya sasa - je kuna evolution?
   - Je, kuna mwenendo au trend wa mada zinazofollowiwa?
   - Impact ya time period kwenye content na approach

5. **AINA ZA VITABU (Book Types & Formats)**
   - Ni types gani za vitabu? (Textbooks, References, Manuals, etc.)
   - Format na structure ya kila kitabu
   - Special features (exercises, illustrations, case studies, etc.)

6. **TARGET AUDIENCE (Wanasomaji Wanaolengwa)**
   - Je, vitabu vimefanywa kwa nani? (Students, Professionals, General public)
   - Difficulty levels zinatofautianaje?
   - Je, kuna vitabu vinavyofaa kwa beginners vs advanced readers?

7. **WACHAPISHAJI (Publishers & Quality)**
   - Wachapishaji ni nani?
   - Je, kuna publishers wanaojirudia?
   - Academic publishers vs commercial publishers
   - Impact ya publisher kwenye quality na credibility

8. **MAUDHUI NA KINA (Content Depth & Coverage)**
   - Kitabu kipi kina comprehensive coverage zaidi?
   - Je, kuna vitabu vinavyogusa surface tu vs vingine vinastudy kwa kina?
   - Breadth vs depth - je kuna balance?

9. **TOFAUTI KUBWA (Major Differences)**
   - Ni nini kinachowafanya vitabu hivi wawe unique?
   - Kitabu kipi kinatofautiana sana na wengine?
   - Je, kuna perspective au angle tofauti?
   - Innovation au uniqueness ya kila kitabu

10. **UFANANI (Similarities & Common Ground)**
    - Je, kuna mambo yanayofanana sana kati ya vitabu?
    - Themes au concepts zinazoendelea kuonekana?
    - Je, waandishi wanakubaliana kwenye principles fulani?
    - Complementary books - je, zinaweza kusomwa pamoja?

11. **USABILITY & PRACTICAL VALUE**
    - Kitabu kipi ni practical zaidi?
    - Je, kuna vitabu vinavyofaa kwa reference vs cover-to-cover reading?
    - Real-world applicability ya content

12. **RECOMMENDATIONS (Mapendekezo)**
    - Kitabu gani ungependekeza kwa nani?
    - Order gani ya kusoma vitabu hivi? (Je, kuna natural progression?)
    - Je, vitabu vinaweza complement each other?

Toa jibu kwa **Kiswahili na Kiingereza mixed** (kama unavyozungumza na Tanzanian developer).
Tumia **bullet points** na **clear subheadings**.
Fanya **in-depth analysis** lakini concise na straight to the point.
Include **specific examples** kutoka kwa vitabu.
"""
        
        try:
            print("   → Performing comparative analysis with AI...")
            
            response = None
            max_retries = 5
            
            # Retry loop for rate limiting
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(comparison_prompt)
                    break  # Success
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(kw in error_msg for kw in ["429", "quota", "rate", "limit"]):
                        wait_time = 20 + (attempt * 10)
                        print(f"   ⏳ Rate limit hit! Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            if response is None:
                return "❌ Comparison analysis failed after multiple retry attempts due to API rate limits."
            
            print("   ✅ Comparative analysis completed!")
            return response.text
            
        except Exception as e:
            return f"❌ Error during comparison: {e}"
    
    def generate_markdown_report(self, df: pd.DataFrame, comparison: str, output_file: str = 'BOOKS_ANALYSIS_REPORT.md'):
        """
        Generate a professional and comprehensive Markdown report
        
        Args:
            df (pd.DataFrame): DataFrame containing all book metadata
            comparison (str): Comparative analysis text
            output_file (str): Name of output markdown file
        """
        print(f"\n📝 Generating comprehensive Markdown report...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # ============= HEADER =============
            f.write("# 📚 COMPREHENSIVE BOOKS ANALYSIS REPORT\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}\n\n")
            f.write(f"**Total Books Analyzed:** {len(df)}\n\n")
            f.write(f"**Analysis Tool:** Gemini AI (Google)\n\n")
            f.write("---\n\n")
            
            # ============= TABLE OF CONTENTS =============
            f.write("## 📑 TABLE OF CONTENTS\n\n")
            f.write("1. [Executive Summary](#1-executive-summary)\n")
            f.write("2. [Books Overview](#2-books-overview)\n")
            f.write("3. [Detailed Book Analysis](#3-detailed-book-analysis)\n")
            f.write("4. [Comparative Analysis](#4-comparative-analysis)\n")
            f.write("5. [Statistical Insights](#5-statistical-insights)\n")
            f.write("6. [Thematic Analysis](#6-thematic-analysis)\n")
            f.write("7. [Recommendations](#7-recommendations)\n")
            f.write("8. [Citations](#8-citations)\n\n")
            f.write("---\n\n")
            
            # ============= EXECUTIVE SUMMARY =============
            f.write("## 1. 📊 EXECUTIVE SUMMARY\n\n")
            
            # Calculate statistics
            total_books = len(df)
            unique_authors = df['author'].nunique()
            unique_publishers = df['publisher'].nunique()
            year_range = f"{df['publication_year'].min()} - {df['publication_year'].max()}" if df['publication_year'].dtype != 'object' else "Various"
            
            # Count subject areas
            subject_areas = df['subject_area'].value_counts()
            
            f.write(f"### Key Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| **Total Books** | {total_books} |\n")
            f.write(f"| **Unique Authors** | {unique_authors} |\n")
            f.write(f"| **Unique Publishers** | {unique_publishers} |\n")
            f.write(f"| **Publication Years** | {year_range} |\n")
            f.write(f"| **Subject Areas** | {len(subject_areas)} |\n\n")
            
            f.write(f"### Subject Area Distribution\n\n")
            for area, count in subject_areas.items():
                if area != 'Unknown':
                    percentage = (count / total_books) * 100
                    f.write(f"- **{area}**: {count} book(s) ({percentage:.1f}%)\n")
            
            f.write("\n---\n\n")
            
            # ============= BOOKS OVERVIEW TABLE =============
            f.write("## 2. 📚 BOOKS OVERVIEW\n\n")
            f.write("| # | Title | Author | Year | Publisher | Type |\n")
            f.write("|---|-------|--------|------|-----------|------|\n")
            
            for idx, row in df.iterrows():
                title = row['title'][:40] + "..." if len(str(row['title'])) > 40 else row['title']
                author = row['author'][:30] + "..." if len(str(row['author'])) > 30 else row['author']
                year = row['publication_year']
                publisher = row['publisher'][:25] + "..." if len(str(row['publisher'])) > 25 else row['publisher']
                book_type = row['book_type']
                
                f.write(f"| {idx + 1} | {title} | {author} | {year} | {publisher} | {book_type} |\n")
            
            f.write("\n---\n\n")
            
            # ============= DETAILED ANALYSIS FOR EACH BOOK =============
            f.write("## 3. 🔍 DETAILED BOOK ANALYSIS\n\n")
            
            for idx, row in df.iterrows():
                f.write(f"### Book {idx + 1}: {row['title']}\n\n")
                
                # Bibliographic Information
                f.write("#### 📖 Bibliographic Information\n\n")
                f.write(f"- **Title:** {row['title']}\n")
                f.write(f"- **Author(s):** {row['author']}\n")
                f.write(f"- **Publication Year:** {row['publication_year']}\n")
                f.write(f"- **Publisher:** {row['publisher']}\n")
                f.write(f"- **Publication Place:** {row['publication_place']}\n")
                f.write(f"- **Edition:** {row['edition']}\n")
                f.write(f"- **ISBN:** {row['isbn']}\n")
                f.write(f"- **Pages:** {row['pages']}\n")
                f.write(f"- **Language:** {row['language']}\n\n")
                
                # Content Information
                f.write("#### 📝 Content Information\n\n")
                f.write(f"- **Book Type:** {row['book_type']}\n")
                f.write(f"- **Subject Area:** {row['subject_area']}\n")
                f.write(f"- **Difficulty Level:** {row['difficulty_level']}\n")
                f.write(f"- **Target Audience:** {row['target_audience']}\n")
                f.write(f"- **Writing Style:** {row['writing_style']}\n")
                f.write(f"- **Approach:** {row['approach']}\n\n")
                
                # Summary
                f.write("#### 📋 Book Summary\n\n")
                f.write(f"{row['book_summary']}\n\n")
                
                # Themes and Concepts
                f.write("#### 🎯 Main Themes\n\n")
                if isinstance(row['main_themes'], list) and row['main_themes']:
                    for theme in row['main_themes']:
                        f.write(f"- {theme}\n")
                else:
                    f.write("- Not available\n")
                f.write("\n")
                
                f.write("#### 💡 Key Concepts\n\n")
                if isinstance(row['key_concepts'], list) and row['key_concepts']:
                    for concept in row['key_concepts']:
                        f.write(f"- {concept}\n")
                else:
                    f.write("- Not available\n")
                f.write("\n")
                
                # Structure
                if row['table_of_contents'] and row['table_of_contents'] != 'Unknown':
                    f.write("#### 📑 Table of Contents\n\n")
                    if isinstance(row['table_of_contents'], list):
                        for chapter in row['table_of_contents']:
                            f.write(f"- {chapter}\n")
                    else:
                        f.write(f"{row['table_of_contents']}\n")
                    f.write("\n")
                
                # Additional Information
                f.write("#### ℹ️ Additional Information\n\n")
                f.write(f"- **Special Features:** {row['special_features']}\n")
                f.write(f"- **Intended Use:** {row['intended_use']}\n")
                f.write(f"- **Series Information:** {row['series_info']}\n")
                
                if isinstance(row['related_fields'], list) and row['related_fields']:
                    f.write(f"- **Related Fields:** {', '.join(row['related_fields'])}\n")
                f.write("\n")
                
                # Preface/Foreword
                if row['preface_summary'] and row['preface_summary'] != 'Unknown':
                    f.write("#### 📄 Preface/Introduction Summary\n\n")
                    f.write(f"{row['preface_summary']}\n\n")
                
                if row['foreword_by'] and row['foreword_by'] != 'Unknown':
                    f.write(f"**Foreword by:** {row['foreword_by']}\n\n")
                
                # Citations
                f.write("#### 📚 Citations\n\n")
                f.write(f"**APA:**\n```\n{row['citation_apa']}\n```\n\n")
                f.write(f"**MLA:**\n```\n{row['citation_mla']}\n```\n\n")
                f.write(f"**Chicago:**\n```\n{row['citation_chicago']}\n```\n\n")
                
                f.write("---\n\n")
            
            # ============= COMPARATIVE ANALYSIS =============
            f.write("## 4. 🔄 COMPARATIVE ANALYSIS\n\n")
            f.write(comparison)
            f.write("\n\n---\n\n")
            
            # ============= STATISTICAL INSIGHTS =============
            f.write("## 5. 📈 STATISTICAL INSIGHTS\n\n")
            
            # Publication Years Distribution
            f.write("### Publication Years Distribution\n\n")
            years = df['publication_year'].value_counts().sort_index()
            f.write("| Year | Number of Books |\n")
            f.write("|------|----------------|\n")
            for year, count in years.items():
                if year != 'Unknown':
                    f.write(f"| {year} | {count} |\n")
            f.write("\n")
            
            # Publishers Analysis
            f.write("### Publishers Analysis\n\n")
            publishers = df['publisher'].value_counts()
            f.write("| Publisher | Books Published |\n")
            f.write("|-----------|----------------|\n")
            for publisher, count in publishers.items():
                if publisher != 'Unknown':
                    f.write(f"| {publisher} | {count} |\n")
            f.write("\n")
            
            # Book Types Distribution
            f.write("### Book Types Distribution\n\n")
            book_types = df['book_type'].value_counts()
            for book_type, count in book_types.items():
                if book_type != 'Unknown':
                    percentage = (count / total_books) * 100
                    f.write(f"- **{book_type}**: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Difficulty Levels
            f.write("### Difficulty Levels\n\n")
            difficulty = df['difficulty_level'].value_counts()
            for level, count in difficulty.items():
                if level != 'Unknown':
                    f.write(f"- **{level}**: {count} book(s)\n")
            f.write("\n")
            
            # Target Audience
            f.write("### Target Audience Analysis\n\n")
            audiences = df['target_audience'].value_counts()
            for audience, count in audiences.items():
                if audience != 'Unknown':
                    f.write(f"- **{audience}**: {count} book(s)\n")
            f.write("\n")
            
            f.write("---\n\n")
            
            # ============= THEMATIC ANALYSIS =============
            f.write("## 6. 🎨 THEMATIC ANALYSIS\n\n")
            
            # Collect all themes
            all_themes = []
            for themes_list in df['main_themes']:
                if isinstance(themes_list, list):
                    all_themes.extend(themes_list)
            
            if all_themes:
                theme_freq = Counter(all_themes).most_common(20)
                
                f.write("### Most Common Themes (Top 20)\n\n")
                f.write("| Rank | Theme | Frequency |\n")
                f.write("|------|-------|----------|\n")
                for rank, (theme, count) in enumerate(theme_freq, 1):
                    f.write(f"| {rank} | {theme} | {count} |\n")
                f.write("\n")
            
            # Collect all key concepts
            all_concepts = []
            for concepts_list in df['key_concepts']:
                if isinstance(concepts_list, list):
                    all_concepts.extend(concepts_list)
            
            if all_concepts:
                concept_freq = Counter(all_concepts).most_common(15)
                
                f.write("### Most Common Key Concepts (Top 15)\n\n")
                for rank, (concept, count) in enumerate(concept_freq, 1):
                    f.write(f"{rank}. **{concept}** - {count} occurrence(s)\n")
                f.write("\n")
            
            f.write("---\n\n")
            
            # ============= RECOMMENDATIONS =============
            f.write("## 7. 💡 RECOMMENDATIONS\n\n")
            
            f.write("### Reading Recommendations\n\n")
            
            # Beginner books
            beginner_books = df[df['difficulty_level'].str.contains('Beginner', case=False, na=False)]
            if not beginner_books.empty:
                f.write("#### For Beginners\n\n")
                for _, book in beginner_books.iterrows():
                    f.write(f"- **{book['title']}** by {book['author']}\n")
                    f.write(f"  - *Why:* {book['book_summary'][:100]}...\n")
                f.write("\n")
            
            # Intermediate/Advanced books
            advanced_books = df[df['difficulty_level'].str.contains('Advanced|Intermediate', case=False, na=False)]
            if not advanced_books.empty:
                f.write("#### For Advanced Readers\n\n")
                for _, book in advanced_books.iterrows():
                    f.write(f"- **{book['title']}** by {book['author']}\n")
                f.write("\n")
            
            f.write("### Reading Order Suggestion\n\n")
            f.write("Based on difficulty levels and content progression:\n\n")
            
            # Sort by difficulty and year
            difficulty_order = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
            df['difficulty_rank'] = df['difficulty_level'].map(difficulty_order)
            sorted_books = df.sort_values(['difficulty_rank', 'publication_year'])
            
            for idx, (_, book) in enumerate(sorted_books.iterrows(), 1):
                f.write(f"{idx}. **{book['title']}** ({book['difficulty_level']}) - {book['publication_year']}\n")
            
            f.write("\n")
            
            f.write("### Cross-Reference Recommendations\n\n")
            f.write("Books that complement each other:\n\n")
            
            # Group by subject area
            for subject in df['subject_area'].unique():
                if subject != 'Unknown':
                    subject_books = df[df['subject_area'] == subject]
                    if len(subject_books) > 1:
                        f.write(f"**{subject}:**\n")
                        for _, book in subject_books.iterrows():
                            f.write(f"- {book['title']}\n")
                        f.write("\n")
            
            f.write("---\n\n")
            
            # ============= CITATIONS =============
            f.write("## 8. 📖 CITATIONS\n\n")
            
            f.write("### APA Format\n\n")
            for idx, row in df.iterrows():
                f.write(f"{idx + 1}. {row['citation_apa']}\n\n")
            
            f.write("### MLA Format\n\n")
            for idx, row in df.iterrows():
                f.write(f"{idx + 1}. {row['citation_mla']}\n\n")
            
            f.write("### Chicago Format\n\n")
            for idx, row in df.iterrows():
                f.write(f"{idx + 1}. {row['citation_chicago']}\n\n")
            
            f.write("---\n\n")
            
            # ============= FOOTER =============
            f.write("## 📌 NOTES\n\n")
            f.write("- This report was automatically generated using AI analysis\n")
            f.write("- All information is extracted from the first 30 pages of each book\n")
            f.write("- Some metadata may be incomplete if not present in the analyzed pages\n")
            f.write("- For academic citations, please verify details with the actual books\n\n")
            
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using Gemini AI*\n")
            f.write(f"*Analyzer: Book Analysis Tool v1.0*\n")
        
        print(f"✅ Comprehensive Markdown report saved: {output_file}")
    
    def analyze_books_folder(self, folder_path: str = 'BOOKS2'):
        """
        Main function to analyze all books in the specified folder
        
        Args:
            folder_path (str): Path to folder containing book PDF files
            
        Returns:
            tuple: (DataFrame, comparison_text) or (None, None) if analysis fails
        """
        print(f"\n{'='*70}")
        print(f"📚 COMPREHENSIVE BOOK ANALYSIS")
        print(f"📁 Folder: {folder_path}")
        print(f"{'='*70}\n")
        
        # Verify folder exists
        if not os.path.exists(folder_path):
            print(f"❌ Folder '{folder_path}' does not exist!")
            print(f"💡 Please create the folder and add your PDF files")
            return None, None
        
        # Get all PDF files from folder
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDF files found in '{folder_path}'!")
            print(f"💡 Please add PDF files to the folder")
            return None, None
        
        print(f"✅ Found {len(pdf_files)} book(s) to analyze")
        print(f"⏱️  Estimated time: ~{len(pdf_files) * 25 // 60} minute(s)")
        print(f"⏰ Analysis started at: {datetime.now().strftime('%H:%M:%S')}\n")
        
        # Process each book
        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"\n{'─'*70}")
            print(f"[{idx}/{len(pdf_files)}] 📚 Analyzing: {pdf_file}")
            print(f"{'─'*70}")
            
            pdf_path = os.path.join(folder_path, pdf_file)
            
            # Step 1: Extract text from PDF
            book_text = self.extract_text_from_pdf(pdf_path)
            
            if not book_text:
                print("   ⚠️ Could not extract text from this PDF, skipping...")
                # Add placeholder with basic info
                self.books_data.append(self._create_default_metadata(pdf_file))
                continue
            
            # Step 2: Extract comprehensive metadata using AI
            metadata = self.extract_book_metadata(book_text, pdf_file)
            self.books_data.append(metadata)
            
            # Display quick summary
            print(f"\n   📝 EXTRACTED INFORMATION:")
            print(f"      Title: {metadata.get('title', 'Unknown')[:60]}...")
            print(f"      Author: {metadata.get('author', 'Unknown')[:50]}...")
            print(f"      Year: {metadata.get('publication_year', 'Unknown')}")
            print(f"      Publisher: {metadata.get('publisher', 'Unknown')[:40]}...")
            print(f"      Type: {metadata.get('book_type', 'Unknown')}")
            print(f"      Subject: {metadata.get('subject_area', 'Unknown')}")
        
        # Verify we have data
        if not self.books_data:
            print("\n❌ No book data could be extracted!")
            return None, None
        
        # Step 3: Create comprehensive DataFrame
        print(f"\n{'='*70}")
        print("📊 CREATING COMPREHENSIVE DATAFRAME...")
        print(f"{'='*70}\n")
        
        df = pd.DataFrame(self.books_data)
        
        print(f"✅ DataFrame created with {len(df)} books and {len(df.columns)} metadata fields\n")
        
        # Step 4: Perform comparative analysis
        print(f"{'='*70}")
        print("🔍 PERFORMING COMPARATIVE ANALYSIS...")
        print(f"⏳ Waiting 25 seconds before comparison (rate limit management)...")
        print(f"{'='*70}\n")
        
        time.sleep(25)  # Longer delay before comprehensive comparison
        comparison = self.compare_books(self.books_data)
        
        return df, comparison


# ============================================
# MAIN EXECUTION SCRIPT
# ============================================

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    API_KEY = "AIzaSyBkfgpmu9IaNNJbVU_4BO8J-577WMOeLHM"  # Your Gemini API key
    BOOKS_FOLDER = "BOOKS2"  # Folder containing your PDF books
    
    # ===== STARTUP =====
    print("="*70)
    print("📚 COMPREHENSIVE BOOK ANALYZER")
    print("="*70)
    print("Powered by Google Gemini AI")
    print("="*70)
    
    # Initialize analyzer
    analyzer = BookAnalyzer(API_KEY)
    
    # Run analysis
    df, comparison = analyzer.analyze_books_folder(BOOKS_FOLDER)
    
    # ===== PROCESS RESULTS =====
    if df is not None:
        # Display summary
        print("\n" + "="*70)
        print("📊 ANALYSIS COMPLETE!")
        print("="*70 + "\n")
        
        print(f"✅ Successfully analyzed {len(df)} book(s)\n")
        
        # Display quick overview
        display_cols = ['title', 'author', 'publication_year', 'book_type', 'subject_area']
        print("Quick Overview:")
        print(df[display_cols].to_string(max_colwidth=30))
        
        # ===== SAVE OUTPUT FILES =====
        print("\n" + "="*70)
        print("💾 SAVING OUTPUT FILES...")
        print("="*70)
        
        # 1. Full CSV file
        csv_file = 'books_complete_analysis.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✅ Complete CSV: {csv_file}")
        
        # 2. Excel workbook with multiple sheets
        try:
            excel_file = 'books_complete_analysis.xlsx'
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Sheet 1: Basic Info
                basic_cols = ['filename', 'title', 'author', 'publication_year', 'publisher', 
                             'publication_place', 'isbn', 'pages']
                df[basic_cols].to_excel(writer, sheet_name='Basic Info', index=False)
                
                # Sheet 2: Content Details
                content_cols = ['title', 'book_type', 'subject_area', 'book_summary', 
                               'target_audience', 'difficulty_level']
                df[content_cols].to_excel(writer, sheet_name='Content', index=False)
                
                # Sheet 3: Style & Approach
                style_cols = ['title', 'writing_style', 'approach', 'special_features', 'intended_use']
                df[style_cols].to_excel(writer, sheet_name='Style', index=False)
                
                # Sheet 4: Citations
                citation_cols = ['title', 'author', 'citation_apa', 'citation_mla', 'citation_chicago']
                df[citation_cols].to_excel(writer, sheet_name='Citations', index=False)
            
            print(f"✅ Excel workbook (4 sheets): {excel_file}")
            
        except Exception as e:
            print(f"⚠️ Excel save error: {e}")
            print(f"   Tip: Install openpyxl with: pip install openpyxl")
        
        # 3. Comparison analysis text file
        comparison_file = 'books_comparative_analysis.txt'
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("COMPARATIVE ANALYSIS OF BOOKS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Books: {len(df)}\n\n")
            f.write("="*70 + "\n\n")
            f.write(comparison)
        print(f"✅ Comparative analysis: {comparison_file}")
        
        # 4. Professional Markdown report
        analyzer.generate_markdown_report(df, comparison)
        
        # ===== DISPLAY COMPARISON =====
        print("\n" + "="*70)
        print("🔍 COMPARATIVE ANALYSIS")
        print("="*70 + "\n")
        print(comparison)
        
        # ===== DISPLAY STATISTICS =====
        print("\n" + "="*70)
        print("📈 STATISTICAL SUMMARY")
        print("="*70)
        
        print(f"\n📚 Total Books: {len(df)}")
        print(f"✍️  Unique Authors: {df['author'].nunique()}")
        print(f"🏢 Unique Publishers: {df['publisher'].nunique()}")
        
        # Years distribution
        years = df['publication_year'].value_counts().sort_index()
        print(f"\n📅 Publication Years:")
        for year, count in years.items():
            if year != 'Unknown':
                print(f"   {year}: {count} book(s)")
        
        # Subject areas
        subjects = df['subject_area'].value_counts()
        print(f"\n📖 Subject Areas:")
        for subject, count in subjects.items():
            if subject != 'Unknown':
                print(f"   {subject}: {count}")
        
        # Book types
        book_types = df['book_type'].value_counts()
        print(f"\n📕 Book Types:")
        for btype, count in book_types.items():
            if btype != 'Unknown':
                print(f"   {btype}: {count}")
        
        # Themes analysis
        all_themes = []
        for themes_list in df['main_themes']:
            if isinstance(themes_list, list):
                all_themes.extend(themes_list)
        
        if all_themes:
            theme_freq = Counter(all_themes).most_common(10)
            print(f"\n🎯 Top 10 Themes:")
            for theme, count in theme_freq:
                print(f"   {theme}: {count}")
        
        # ===== FINAL SUMMARY =====
        print("\n" + "="*70)
        print("✅ ANALYSIS SUCCESSFULLY COMPLETED!")
        print("="*70)
        print(f"⏰ Finished at: {datetime.now().strftime('%H:%M:%S')}\n")
        
        print("📁 OUTPUT FILES GENERATED:")
        print("   1. books_complete_analysis.csv - Full data in CSV")
        print("   2. books_complete_analysis.xlsx - Excel with 4 sheets")
        print("   3. books_comparative_analysis.txt - Text comparison")
        print("   4. BOOKS_ANALYSIS_REPORT.md - Professional Markdown report")
        print("\n💡 Open the .md file in a Markdown viewer for best experience!")
        
    else:
        print("\n❌ Analysis failed. Please check:")
        print("   1. Folder 'BOOKS2' exists in current directory")
        print("   2. PDF files are present in the folder")
        print("   3. PDFs are readable (not corrupted)")
        print("   4. API key is valid and has remaining quota")
        print("\n💡 For rate limit issues, wait a few minutes and try again")
```

    ======================================================================
    📚 COMPREHENSIVE BOOK ANALYZER
    ======================================================================
    Powered by Google Gemini AI
    ======================================================================
    🔑 Configuring API key...
    ✅ API connection successful! Available models: 54
    ✅ Using model: gemini-2.5-flash
    ⚠️  FREE PLAN MODE: 20-second delays between API calls
    
    
    ======================================================================
    📚 COMPREHENSIVE BOOK ANALYSIS
    📁 Folder: BOOKS2
    ======================================================================
    
    ✅ Found 3 book(s) to analyze
    ⏱️  Estimated time: ~1 minute(s)
    ⏰ Analysis started at: 13:24:00
    
    
    ──────────────────────────────────────────────────────────────────────
    [1/3] 📚 Analyzing: Essentials of Organisational Behaviour by Laurie J. Mullins (z-lib.org).pdf
    ──────────────────────────────────────────────────────────────────────
       → Total pages in book: 523
       → Extracting text from first 30 pages...
       ✅ Extracted 85872 characters
       → Analyzing book with Gemini AI...
       ✅ Metadata successfully extracted!
       ⏳ Cooling down for 20 seconds...
    
       📝 EXTRACTED INFORMATION:
          Title: ESSENTIALS OF ORGANISATIONAL BEHAVIOUR...
          Author: Laurie J. Mullins, Gill Christy...
          Year: 2011
          Publisher: Pearson Education Limited...
          Type: Textbook
          Subject: Organisational Behaviour
    
    ──────────────────────────────────────────────────────────────────────
    [2/3] 📚 Analyzing: Organisational behaviour by Kinicki, Angelo Kreitner, Robert Sinding, Knud Waldstrøm, Christian (z-lib.org).pdf
    ──────────────────────────────────────────────────────────────────────
       → Total pages in book: 691
       → Extracting text from first 30 pages...
       ✅ Extracted 46205 characters
       → Analyzing book with Gemini AI...
       ✅ Metadata successfully extracted!
       ⏳ Cooling down for 20 seconds...
    
       📝 EXTRACTED INFORMATION:
          Title: Organisational Behaviour...
          Author: Knud Sinding, Christian Waldstrom...
          Year: 2014
          Publisher: McGraw-Hill Education...
          Type: Textbook
          Subject: Organisational Behaviour
    
    ──────────────────────────────────────────────────────────────────────
    [3/3] 📚 Analyzing: Organisational behaviour individuals, groups and organisation by Brooks, Ian (z-lib.org)(1).pdf
    ──────────────────────────────────────────────────────────────────────
       → Total pages in book: 374
       → Extracting text from first 30 pages...
       ✅ Extracted 36469 characters
       → Analyzing book with Gemini AI...
       ✅ Metadata successfully extracted!
       ⏳ Cooling down for 20 seconds...
    
       📝 EXTRACTED INFORMATION:
          Title: Organisational Behaviour: Individuals, Groups and Organisati...
          Author: Ian Brooks, Hugh Davenport, Jon Stephens, Stephen ...
          Year: 2009
          Publisher: Pearson Education Limited...
          Type: Textbook
          Subject: Organisational Behaviour
    
    ======================================================================
    📊 CREATING COMPREHENSIVE DATAFRAME...
    ======================================================================
    
    ✅ DataFrame created with 3 books and 30 metadata fields
    
    ======================================================================
    🔍 PERFORMING COMPARATIVE ANALYSIS...
    ⏳ Waiting 25 seconds before comparison (rate limit management)...
    ======================================================================
    
       → Performing comparative analysis with AI...
       ✅ Comparative analysis completed!
    
    ======================================================================
    📊 ANALYSIS COMPLETE!
    ======================================================================
    
    ✅ Successfully analyzed 3 book(s)
    
    Quick Overview:
                               title                         author publication_year book_type              subject_area
    0  ESSENTIALS OF ORGANISATION...  Laurie J. Mullins, Gill Ch...             2011  Textbook  Organisational Behaviour
    1       Organisational Behaviour  Knud Sinding, Christian Wa...             2014  Textbook  Organisational Behaviour
    2  Organisational Behaviour: ...  Ian Brooks, Hugh Davenport...             2009  Textbook  Organisational Behaviour
    
    ======================================================================
    💾 SAVING OUTPUT FILES...
    ======================================================================
    ✅ Complete CSV: books_complete_analysis.csv
    ✅ Excel workbook (4 sheets): books_complete_analysis.xlsx
    ✅ Comparative analysis: books_comparative_analysis.txt
    
    📝 Generating comprehensive Markdown report...
    ✅ Comprehensive Markdown report saved: BOOKS_ANALYSIS_REPORT.md
    
    ======================================================================
    🔍 COMPARATIVE ANALYSIS
    ======================================================================
    
    Hii ni detailed comparative analysis ya vitabu vitatu vya Organisational Behaviour, kama ulivyoagiza.
    
    ***
    
    ### Comprehensive Comparison ya Vitabu vya Organisational Behaviour
    
    Hapa chini kuna uchambuzi wa kina kulinganisha vitabu hivi vitatu muhimu katika eneo la Organisational Behaviour (OB).
    
    ---
    
    #### 1. MADA KUU (Main Topics & Subject Areas)
    
    *   **Ufanani:** Vitabu vyote vitatu vinashughulikia **core themes** za Organisational Behaviour. Kwa ujumla, vinagawanyika katika maeneo makuu matatu:
        *   **Individual Processes:** Kila kitabu kinazungumzia mada kama vile personality, values, attitudes, emotions, perception, communication, na motivation.
        *   **Group and Social Processes:** Zinajumuisha group dynamics, teamwork, leadership, na stress.
        *   **Organisational Processes:** Zinaangalia organisational strategy, structure, culture, power, politics, conflict, na change management.
        *   **Subject Area Dominant:** Ni wazi kuwa **Organisational Behaviour** ndio subject area kuu inayodominika katika vitabu vyote.
    
    *   **Tofauti na Msisitizo:**
        *   **"ESSENTIALS OF ORGANISATIONAL BEHAVIOUR" (Mullins & Christy):** Kichwa chake "Essentials" kinaonyesha msisitizo kwenye mada za msingi. Inatoa "concise introduction" na inalenga "managerial behaviour." Themes zake zimepangiliwa vizuri kuanzia "Organisational Setting" hadi "Organisational Culture and Change," ikionyesha mlolongo wa kueleweka.
        *   **"Organisational Behaviour" (Sinding & Waldstrom):** Hiki ni "comprehensive textbook" na kinaeleza "meticulously explores individual, group, and organisational processes." Inajumuisha mada za ziada kama "Foundations of organisational behaviour and research methods," "International and global cultural differences," na "Effectiveness and decline of organisations," kuonyesha breadth na depth zaidi.
        *   **"Organisational Behaviour: Individuals, Groups and Organisation" (Brooks et al.):** Inatoa "comprehensive introduction" na inaweka msisitizo kwenye "human behaviour within organisational contexts," na "applied perspective" inayoendana na behavioral sciences. Pia inajumuisha themes kama "Diversity, change, conflict, and communication" zikiwa integrated throughout, badala ya kuwa sura tofauti tu.
    
    *   **Progression:** Hakuna progression ya moja kwa moja ambapo kitabu kimoja kinajenga juu ya kingine kwa mpangilio wa mada. Kila kitabu kimekusudiwa kutoa msingi kamili. Hata hivyo, Sinding kinaonekana kuwa na mada nyingi na za kina zaidi, kikienda mbali zaidi ya "essentials" za Mullins au "introduction" ya Brooks, hasa kwa kujumuisha research methods na international differences kwa kina.
    
    ---
    
    #### 2. WAANDISHI (Authors & Their Backgrounds)
    
    *   **Idadi na Kurudiwa:**
        *   **Mullins & Christy:** Waandishi wawili (Laurie J. Mullins, Gill Christy).
        *   **Sinding & Waldstrom:** Waandishi wawili (Knud Sinding, Christian Waldstrom).
        *   **Brooks et al.:** Waandishi wanne (Ian Brooks, Hugh Davenport, Jon Stephens, Stephen Swailes).
        *   Hakuna mwandishi anayeonekana mara nyingi zaidi ya kitabu kimoja kati ya hivi vitatu.
    
    *   **Background na Credibility:**
        *   Kwa kuzingatia aina ya vitabu ("Textbook"), wachapishaji (Pearson, McGraw-Hill), na maudhui (academic, theoretical, practical), ni salama kusema waandishi wote wana **backgrounds za kitaaluma (academic)**. Mara nyingi, waandishi wa vitabu kama hivi ni maprofesa au watafiti katika fields za Organisational Behaviour, Management, au Business Studies.
        *   **Credibility na expertise** ya waandishi wote ni ya juu kutokana na:
            *   Kuchapishwa na wachapishaji mashuhuri wa vitabu vya kielimu.
            *   Kueleza dhana za kina na mifano ya vitendo.
            *   Vitabu vingine kuwa na "editions" nyingi (Mullins is the 3rd, Sinding is the 5th), ikionyesha kuwa kazi zao zimepokelewa vizuri na zimesasishwa mara kwa mara, ambacho ni kiashiria cha expertise in the field.
    
    ---
    
    #### 3. MITINDO YA UANDISHI (Writing Styles & Approaches)
    
    *   **Ufanani:** Vitabu vyote vina mtindo wa **Academic** na **Theoretical with Practical Applications**. Zote zimeundwa kwa lengo la kufundisha na kuelimisha wanafunzi wa chuo kikuu.
    
    *   **Tofauti:**
        *   **"ESSENTIALS OF ORGANISATIONAL BEHAVIOUR" (Mullins & Christy):**
            *   **Style:** "Academic, detailed, engaging, and concise." Neno "engaging" linaweza kuashiria mtindo unaojaribu kuunganisha msomaji zaidi, na "concise" linaashiria kuwa linafupisha mada muhimu bila kuzama sana kwenye undani usiohitajika kwa utangulizi.
            *   **Approach:** "A blend of theoretical exposition... and practical application (real-life examples, case studies)." Msisitizo uko kwenye kutoa insights kutoka kwa "experienced teacher's insights."
            *   **Tone:** Formal, but possibly slightly more conversational or approachable due to "engaging" and "concise" nature.
    
        *   **"Organisational Behaviour" (Sinding & Waldstrom):**
            *   **Style:** "Academic, structured, and comprehensive, suitable for a university-level textbook." Hii inaashiria mtindo wa moja kwa moja, uliopangiliwa, na uliojaa maudhui mengi.
            *   **Approach:** "Theoretical and conceptual foundations integrated with practical applications, research evidence, and numerous case studies. Includes experiential learning." Inajumuisha "research evidence" ambayo huenda isisisitizwe sana kwenye vitabu vingine, ikionyesha kina cha kitaaluma.
            *   **Tone:** Formal and technical, characteristic of a comprehensive university textbook.
    
        *   **"Organisational Behaviour: Individuals, Groups and Organisation" (Brooks et al.):**
            *   **Style:** "Academic, succinct, focused, and accessible, designed to be less daunting than larger, North American-origin texts." Msisitizo hapa ni kwenye "accessibility" na kufanya kitabu kisisumbue kusoma.
            *   **Approach:** "Theoretical with practical applications, presented as an applied behavioural science. Incorporates current research and debates, with a strong emphasis on real-world managerial implications." Neno "applied behavioural science" linaonyesha mwelekeo wa kutumia principles za sayansi ya tabia katika mazingira ya kazi.
            *   **Tone:** Formal, yet aiming for approachability.
    
    *   **Complexity:**
        *   **Sinding & Waldstrom** inaonekana kuwa complex zaidi kutokana na "comprehensive" nature yake na kujumuisha "research evidence" na "experiential learning," na difficulty level ya "Intermediate."
        *   **Mullins & Christy** na **Brooks et al.** zote zina "Beginner to Intermediate" na zinalenga kuwa "concise," "engaging," na "accessible," hivyo zitakuwa easy to read zaidi kwa beginners.
    
    ---
    
    #### 4. MIAKA YA UCHAPISHAJI (Publication Timeline & Trends)
    
    *   **Timeline:**
        *   **Brooks et al.:** 2009 (cha zamani zaidi)
        *   **Mullins & Christy:** 2011
        *   **Sinding & Waldstrom:** 2014 (cha hivi karibuni zaidi)
    
    *   **Evolution na Impact ya Time Period:**
        *   Kutokana na vitabu hivi kuangukia ndani ya miaka 5 (2009-2014), tofauti katika maudhui kutokana na muda haziwezi kuwa kubwa sana kama ingekuwa kuna gap ya miaka 10+ au 20+.
        *   Hata hivyo, kitabu cha **Sinding & Waldstrom (2014)**, kikiwa cha karibuni zaidi na pia kikiwa "Fifth Edition," kina uwezekano mkubwa wa kujumuisha **mwenendo wa sasa (current trends)** na **utafiti wa hivi karibuni (latest research)** katika OB, ikilinganishwa na vitabu vya 2009 na 2011.
        *   Mada kama "diversity," "global cultural differences," na "change management" ni muhimu na huenda zimesasishwa zaidi katika toleo la karibuni, ingawa zote zinaguswa na vitabu vyote. Vitabu vya zamani bado vina umuhimu kwa kutoa misingi imara.
    
    ---
    
    #### 5. AINA ZA VITABU (Book Types & Formats)
    
    *   **Book Type:** Vitabu vyote vitatu ni **"Textbook"**. Hii inamaanisha vimeundwa mahsusi kwa ajili ya kufundishia katika ngazi ya chuo kikuu.
    
    *   **Format na Structure:**
        *   Kama textbooks, zote zitakuwa na **structured format** na sura zilizopangiliwa vizuri.
        *   **Special Features (Common):**
            *   **Case studies:** Vitabu vyote vinavitumia kuonyesha practical application. Mullins ana "exclusive new case studies," Brooks anamalizia na "case studies," na Sinding ana "case studies" nyingi.
            *   **Review Questions / Exercises:** Sinding anavitaja waziwazi ("review questions, and exercises"), na Brooks anataja "personal awareness and group exercises." Hizi ni muhimu kwa kujitathmini na kujifunza kwa vitendo.
            *   **Real-life examples:** Zote zinavitumia kuunganisha nadharia na matumizi halisi.
    
    *   **Special Features (Unique/Emphasized):**
        *   **Mullins:** "experienced teacher's insights" na "exclusive new case studies."
        *   **Sinding:** "integrating theory, evidence, and practice" na "experiential learning through personal awareness and group exercises."
        *   **Brooks:** "applied perspective," "current research and debates," na "real-world managerial implications."
    
    ---
    
    #### 6. TARGET AUDIENCE (Wanasomaji Wanaolengwa)
    
    *   **Ufanani:** Zote zinalenga **students studying organisational behaviour, management, or related business subjects** katika ngazi ya chuo kikuu.
    
    *   **Tofauti za kina na Difficulty Levels:**
        *   **"ESSENTIALS OF ORGANISATIONAL BEHAVIOUR" (Mullins & Christy):**
            *   **Audience:** "Undergraduate students."
            *   **Difficulty:** "Beginner to Intermediate." Hiki kinafaa kwa wanafunzi wanaoanza kusoma OB au wanaohitaji utangulizi mfupi na wa kueleweka.
    
        *   **"Organisational Behaviour" (Sinding & Waldstrom):**
            *   **Audience:** "Undergraduate and postgraduate students... professionals." Hiki kina spectrum pana zaidi, kikiwafaa wanafunzi wa advanced na hata wataalamu wanaotaka kuongeza uelewa wao.
            *   **Difficulty:** "Intermediate." Hiki kinahitaji kiwango fulani cha uelewa wa awali au utayari wa kujifunza kwa kina.
    
        *   **"Organisational Behaviour: Individuals, Groups and Organisation" (Brooks et al.):**
            *   **Audience:** "Undergraduate students (Level 1 or 2)... postexperience, postgraduate, and professional courses needing an introduction or foundational understanding... general readers." Hiki kina audience pana sana, kikiwa kinafaa kwa wanafunzi wa kuanzia na wale wanaohitaji foundation katika maeneo mengine (HRM, change management) na hata wasomaji wa kawaida.
            *   **Difficulty:** "Beginner to Intermediate." Kimeundwa kuwa "accessible" na "less daunting," kwa hiyo kinafaa sana kwa beginners.
    
    *   **Beginners vs Advanced:**
        *   Kwa **Beginners**: Mullins & Christy na Brooks et al. ni chaguo bora. Mullins ni "concise," na Brooks ni "accessible" na "succinct."
        *   Kwa **Intermediate/Advanced readers na Professionals**: Sinding & Waldstrom ndio chaguo bora kutokana na ukamilifu na kina chake.
    
    ---
    
    #### 7. WACHAPISHAJI (Publishers & Quality)
    
    *   **Wachapishaji:**
        *   **Pearson Education Limited:** Inachapisha vitabu viwili (Mullins & Christy, na Brooks et al.).
        *   **McGraw-Hill Education:** Inachapisha kitabu kimoja (Sinding & Waldstrom).
    
    *   **Kurudiwa na Aina:**
        *   **Pearson Education Limited** na **McGraw-Hill Education** zote ni **leading academic publishers** za vitabu vya elimu ya juu duniani.
        *   Zote ni **commercial publishers** lakini zina reputation kubwa katika soko la vitabu vya elimu na research.
    
    *   **Impact kwenye Quality na Credibility:**
        *   Uchapishaji na nyumba hizi mbili unahakikisha **high quality** ya maudhui, editing, na production.
        *   Wachapishaji hawa hupitia **rigorous peer review processes** kabla ya kuchapisha vitabu, hivyo huongeza **credibility** kubwa kwa waandishi na maudhui.
        *   Kuona Pearson akirudia kunachapisha vitabu viwili kunaonyesha commitment yao katika eneo la Organisational Behaviour na uwezo wao wa kutoa vitabu tofauti vinavyolenga soko tofauti (e.g., "Essentials" vs "Comprehensive Introduction").
    
    ---
    
    #### 8. MAUDHUI NA KINA (Content Depth & Coverage)
    
    *   **Comprehensive Coverage:**
        *   **Sinding & Waldstrom (2014)** ndicho kinachoonekana kuwa na **comprehensive coverage zaidi**. Kinaeleza "meticulously explores individual, group, and organisational processes, integrating theory, evidence, and practice." Kujumuisha "Foundations of organisational behaviour and research methods" na "International and global cultural differences" kunathibitisha kina chake.
        *   **Brooks et al. (2009)** inatoa "comprehensive introduction" na inajumuisha "current research and debates," ikionyesha kina cha kutosha kwa utangulizi.
        *   **Mullins & Christy (2011)** inatoa "concise introduction" na "thoroughly explores core topics." Ingawa inaeleza kwa kina mada za msingi, labda haifiki undani wa Sinding katika kila nyanja.
    
    *   **Surface vs In-depth:**
        *   **Mullins & Christy:** Inaweza kusemwa inagusa "surface" kwa maana ya kutoa "essentials" na kuwa "concise," lakini bado "thoroughly explores core topics," hivyo haiko juu juu kabisa. Inatoa msingi imara.
        *   **Brooks et al.:** Inatoa "applied behavioural science" na "managerial implications," ikijaribu kusawazisha nadharia na matumizi. Kina chake ni cha kutosha kwa utangulizi wa kina.
        *   **Sinding & Waldstrom:** Ndicho kinachostudy kwa **kina zaidi**, hasa kwa kujumuisha "research evidence" na "experiential learning," na kulenga "postgraduate students and professionals."
    
    *   **Breadth vs Depth:**
        *   **Sinding & Waldstrom:** Inaonekana kuwa na balance nzuri ya **breadth na depth**, ikishughulikia mada nyingi kwa kina.
        *   **Mullins & Christy:** Inalenga **depth ya core topics** ndani ya "concise" framework, ikipendelea kina kidogo kwenye mada muhimu kuliko breadth kubwa.
        *   **Brooks et al.:** Inalenga **breadth ya kutosha kwa utangulizi kamili** huku ikitoa "applied perspective," ikijaribu kufanya mada ziwe za kina cha kueleweka.
    
    ---
    
    #### 9. TOFAUTI KUBWA (Major Differences)
    
    *   **Sinding & Waldstrom (2014):**
        *   **Uniqueness:** Comprehensive nature yake, kujumuisha "research evidence," "experiential learning," na kulenga "postgraduate students and professionals." Inatoa mtazamo mpana zaidi (ikiwemo international/global differences) na kina kikubwa, ikiwa ni Fifth Edition. Ni kitabu cha kisasa zaidi.
        *   **Perspective:** Inajumuisha "theory, evidence, and practice" kwa usawa, ikitoa mtazamo wa kisayansi zaidi (evidence-based).
    
    *   **ESSENTIALS OF ORGANISATIONAL BEHAVIOUR (Mullins & Christy, 2011):**
        *   **Uniqueness:** Msisitizo kwenye "essentials" na "concise introduction" inayofundisha misingi kwa ufanisi. Ina "experienced teacher's insights" na "exclusive new case studies," ikilenga kutoa uelewa wa kutosha kwa undergraduate.
        *   **Perspective:** Mwelekeo wa "managerial behaviour," kusaidia wanafunzi kuelewa jinsi nadharia zinavyotumika katika uongozi wa kila siku.
    
    *   **Organisational Behaviour: Individuals, Groups and Organisation (Brooks et al., 2009):**
        *   **Uniqueness:** Inajitofautisha kwa kuwa "less daunting than larger, North American-origin texts" na inatoa "applied perspective" kutoka behavioral sciences. Ni kitabu kinachopatikana na kinazingatia "real-world managerial implications."
        *   **Perspective:** Msisitizo kwenye "human behaviour" na "applied behavioural science," ikijaribu kuunganisha psychology na sociology na mazingira ya shirika.
    
    ---
    
    #### 10. UFANANI (Similarities & Common Ground)
    
    *   **Aina ya Kitabu:** Zote ni **textbooks** za Organisational Behaviour zinazolenga wanafunzi wa chuo kikuu.
    *   **Mada Kuu:** Zote zinafunika **core concepts** za OB, zikianza na masuala ya **individual behaviour**, kisha **group dynamics**, na hatimaye **organisational structures and culture**.
    *   **Approach:** Zote zinatumia mbinu ya **theory na practical application**, zikijumuisha **case studies** na **real-life examples** ili kusaidia uelewa.
    *   **Lengo:** Lengo kuu la vitabu vyote ni **kuwapa wanafunzi uelewa wa kina** wa masuala ya OB na kuwawezesha kutumia principles hizo.
    *   **Themes Zinazojirudia:** Mada kama leadership, motivation, communication, power, conflict, na organisational culture zinapatikana katika themes za vitabu vyote.
    *   **Complementary Books:** Kwa hakika, vitabu hivi vinaweza **kusomwa pamoja** na vinaweza kukamilishana. Kitabu cha Mullins au Brooks kinaweza kutumika kama utangulizi rahisi, kisha Sinding kinaweza kutumika kwa kina zaidi.
    
    ---
    
    #### 11. USABILITY & PRACTICAL VALUE
    
    *   **Practical Value (Overall):** Vitabu vyote vina **practical value** kubwa kwa sababu vinatumiwa kufundisha jinsi watu na mashirika yanavyofanya kazi katika ulimwengu halisi.
    
    *   **Kiwango cha Practicality:**
        *   **"Organisational Behaviour" (Sinding & Waldstrom):** Ni **practical sana** kutokana na kujumuisha "experiential learning through personal awareness and group exercises" na "integrating theory, evidence, and practice." Hiki kinaweza kutumika vizuri kwa kujifunza kwa vitendo darasani au katika training kwa wataalamu.
        *   **"Organisational Behaviour: Individuals, Groups and Organisation" (Brooks et al.):** Hiki pia ni **practical sana** kwa msisitizo wake kwenye "applied perspective" na "real-world managerial implications." Ni nzuri kwa reference kwa maamuzi ya usimamizi.
        *   **"ESSENTIALS OF ORGANISATIONAL BEHAVIOUR" (Mullins & Christy):** Ni **practical** kwa kutoa "real-life examples and exclusive new case studies." Ni nzuri kwa kuelewa jinsi nadharia zinavyotumika katika scenarios za kawaida za biashara.
    
    *   **Reference vs Cover-to-Cover Reading:**
        *   **Mullins & Christy:** Bora kwa **cover-to-cover reading** kwa beginners na kama **concise reference** kwa mada muhimu.
        *   **Brooks et al.:** Nzuri kwa **cover-to-cover reading** kwa utangulizi na pia kama **reference** kwa managerial implications.
        *   **Sinding & Waldstrom:** Inaweza kusomwa **cover-to-cover** kwa uelewa wa kina na pia kama **comprehensive reference** kwa wanafunzi wa advanced na professionals.
    
    ---
    
    #### 12. RECOMMENDATIONS (Mapendekezo)
    
    *   **Kitabu Gani kwa Nani:**
        *   **Kwa Beginners (Wanafunzi wa mwaka wa kwanza au wale wanaohitaji utangulizi rahisi):** Ningependekeza **"ESSENTIALS OF ORGANISATIONAL BEHAVIOUR" by Mullins & Christy** au **"Organisational Behaviour: Individuals, Groups and Organisation" by Brooks et al.** Vitabu hivi vimeundwa kuwa accessible, concise, na vinatoa msingi imara bila kuwa over-complex. Brooks kinaweza kuwa bora zaidi kwa sababu kinajitangaza kuwa "less daunting."
        *   **Kwa Intermediate Students (Wanafunzi wa mwaka wa pili au wa tatu), Postgraduate Students, na Professionals:** Ningependekeza **"Organisational Behaviour" by Sinding & Waldstrom**. Kitabu hiki kina coverage pana na kina, kimejengwa juu ya "evidence," na kinafaa kwa wale wanaotafuta uelewa wa kina na experiential learning.
    
    *   **Order ya Kusoma (Natural Progression):**
        1.  **Start with either Mullins & Christy (2011) or Brooks et al. (2009):** Hivi vitakupa msingi imara na utangulizi wa OB kwa njia inayoeleweka na isiyochosha. Brooks hasa inajulikana kwa accessibility yake.
        2.  **Then progress to Sinding & Waldstrom (2014):** Baada ya kuelewa misingi, kitabu cha Sinding kitakupa fursa ya kuzama kwa kina, kujumuisha utafiti, na kujifunza kwa vitendo, na pia kurekebisha uelewa wako na dhana za kisasa zaidi.
    
    *   **Vinaweza Kukamilishana (Complement Each Other)?**
        *   **Ndio, kabisa!** Vitabu hivi vinaweza kukamilishana vizuri sana. Kuanza na kitabu cha "essentials" au "introduction" (Mullins au Brooks) huweka foundation. Kisha, kutumia kitabu "comprehensive" (Sinding) huruhusu msomaji kupata undani zaidi, mifano ya ziada, na mitazamo mipana zaidi, ikiwemo utafiti wa kisasa. Kwa mfumo wa elimu, mwalimu anaweza kutumia "essentials" kama kitabu kikuu na vitabu vingine kama rasilimali za ziada au kwa modules za advanced.
    
    ---
    
    ======================================================================
    📈 STATISTICAL SUMMARY
    ======================================================================
    
    📚 Total Books: 3
    ✍️  Unique Authors: 3
    🏢 Unique Publishers: 2
    
    📅 Publication Years:
       2009: 1 book(s)
       2011: 1 book(s)
       2014: 1 book(s)
    
    📖 Subject Areas:
       Organisational Behaviour: 3
    
    📕 Book Types:
       Textbook: 3
    
    🎯 Top 10 Themes:
       The Organisational Setting: 1
       Individual Differences and Diversity: 1
       Perception and Communication: 1
       Work Motivation and Job Satisfaction: 1
       Work Groups, Teams, and Leadership: 1
       Organisational Strategy and Structure: 1
       Control, Power, and Ethics in Organizations: 1
       Organisational Culture and Change: 1
       Foundations of organisational behaviour and research methods: 1
       Individual processes (personality, values, attitudes, emotions, perception, communication, motivation): 1
    
    ======================================================================
    ✅ ANALYSIS SUCCESSFULLY COMPLETED!
    ======================================================================
    ⏰ Finished at: 13:28:54
    
    📁 OUTPUT FILES GENERATED:
       1. books_complete_analysis.csv - Full data in CSV
       2. books_complete_analysis.xlsx - Excel with 4 sheets
       3. books_comparative_analysis.txt - Text comparison
       4. BOOKS_ANALYSIS_REPORT.md - Professional Markdown report
    
    💡 Open the .md file in a Markdown viewer for best experience!
    


```python

```


```python

```
